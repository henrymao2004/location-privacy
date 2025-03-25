import tensorflow as tf
import keras
import numpy as np
import random
from rl_components import CriticNetwork, RewardFunction, PPOAgent

random.seed(2020)
np.random.seed(2020)
tf.random.set_random_seed(2020)

from keras.layers import Input, Add, Average, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding, LayerNormalization, MultiHeadAttention, Dropout
from keras.initializers import he_uniform
from keras.regularizers import l1

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from losses import d_bce_loss, trajLoss

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class LSTM_TrajGAN():
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor, tul_classifier=None):
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        self.keys = keys
        self.vocab_size = vocab_size
        
        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        
        self.x_train = None
        
        # Define the optimizer
        self.optimizer = Adam(0.001, 0.5)

        # Build the trajectory generator
        self.generator = self.build_generator()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=d_bce_loss(gen_trajs[4]), optimizer=self.optimizer, metrics=['accuracy'])

        # Build the critic network for RL
        self.critic = CriticNetwork(max_length, vocab_size).model
        
        # Initialize reward function with TUL classifier
        self.reward_function = RewardFunction(self.discriminator, tul_classifier)
        
        # Initialize PPO agent
        self.ppo_agent = PPOAgent(self.generator, self.critic, self.reward_function)

        # The combined model only trains the trajectory generator
        self.discriminator.trainable = False

        # The discriminator takes generated trajectories as input and makes predictions
        pred = self.discriminator(gen_trajs[:4])

        # The combined model (combining the generator and the discriminator)
        self.combined = Model(inputs=inputs, outputs=pred)
        self.combined.compile(loss=trajLoss(inputs, gen_trajs), optimizer=self.optimizer)
        
        # Save model architectures
        C_model_json = self.combined.to_json()
        with open("params/C_model.json", "w") as json_file:
            json_file.write(C_model_json)
            
        G_model_json = self.generator.to_json()
        with open("params/G_model.json", "w") as json_file:
            json_file.write(G_model_json)
        
        D_model_json = self.discriminator.to_json()
        with open("params/D_model.json", "w") as json_file:
            json_file.write(D_model_json)
            
        Critic_model_json = self.critic.to_json()
        with open("params/Critic_model.json", "w") as json_file:
            json_file.write(Critic_model_json)

    def build_discriminator(self):
        # Input Layer
        inputs = []
        
        # Embedding Layer
        embeddings = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            if key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]),
                          name='input_' + key)

                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)

            else:
                i = Input(shape=(self.max_length,self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=self.vocab_size[key], use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
            
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        d = Dense(units=100, use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1), name='emb_trajpoint')
        dense_outputs = [d(x) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
        
        # Transformer Modeling Layer
        transformer_block = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)
        transformer_output = transformer_block(emb_traj)
        
        # Global average pooling to reduce sequence dimension
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(transformer_output)
        
        # Output
        sigmoid = Dense(1, activation='sigmoid')(avg_pool)

        return Model(inputs=inputs, outputs=sigmoid)

    def build_generator(self):
        # Input Layer
        inputs = []
        
        # Embedding Layer
        embeddings = []
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        mask = Input(shape=(self.max_length, 1), name='input_mask')
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, activation='relu', use_bias=True, kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length,self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=self.vocab_size[key], activation='relu', use_bias=True, kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        inputs.append(noise)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        d = Dense(units=100, use_bias=True, activation='relu', kernel_initializer=he_uniform(seed=1), name='emb_trajpoint')
        dense_outputs = [d(Concatenate(axis=1)([x, noise])) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)
        
        # Transformer Modeling Layer
        transformer_block = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)
        transformer_output = transformer_block(emb_traj)
        
        # Outputs
        outputs = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                output_mask = Lambda(lambda x: x)(mask)
                outputs.append(output_mask)
            elif key == 'lat_lon':
                output = TimeDistributed(Dense(2, activation='tanh'), name='output_latlon')(transformer_output)
                scale_factor = self.scale_factor
                output_stratched = Lambda(lambda x: x * scale_factor)(output)
                outputs.append(output_stratched)
            else:
                output = TimeDistributed(Dense(self.vocab_size[key], activation='softmax'), name='output_' + key)(transformer_output)
                outputs.append(output)
                
        return Model(inputs=inputs, outputs=outputs)

    def train(self, epochs=200, batch_size=256, sample_interval=10, rl_update_interval=5):
        # Training data
        x_train = np.load('data/final_train.npy',allow_pickle=True)
        self.x_train = x_train

        # Padding zero to reach the maxlength
        X_train = [pad_sequences(f, self.max_length, padding='pre', dtype='float64') for f in x_train]
        self.X_train = X_train
        
        for epoch in range(1,epochs+1):
            # Select a random batch of real trajectories
            idx = np.random.randint(0, X_train[0].shape[0], batch_size)
            
            # Ground truths for real trajectories and synthetic trajectories
            real_bc = np.ones((batch_size, 1))
            syn_bc = np.zeros((batch_size, 1))

            real_trajs_bc = []
            real_trajs_bc.append(X_train[0][idx]) # latlon
            real_trajs_bc.append(X_train[1][idx]) # day
            real_trajs_bc.append(X_train[2][idx]) # hour
            real_trajs_bc.append(X_train[3][idx]) # category
            real_trajs_bc.append(X_train[4][idx]) # mask
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            real_trajs_bc.append(noise) # noise

            # Generate a batch of synthetic trajectories
            gen_trajs_bc = self.generator.predict(real_trajs_bc)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_trajs_bc[:4], real_bc)
            d_loss_syn = self.discriminator.train_on_batch(gen_trajs_bc[:4], syn_bc)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_syn)

            # Train the generator with GAN objective
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            real_trajs_bc[5] = noise
            g_loss = self.combined.train_on_batch(real_trajs_bc, real_bc)
            
            # RL updates every rl_update_interval epochs
            if epoch % rl_update_interval == 0:
                # Get current policy probabilities
                old_probs = self.generator.predict(real_trajs_bc)
                
                # Generate trajectories and compute rewards
                gen_trajs = self.generator.predict(real_trajs_bc)
                rewards = []
                for i in range(batch_size):
                    reward = self.reward_function.compute_total_reward(
                        [t[i] for t in gen_trajs], 
                        [t[i] for t in real_trajs_bc[:4]], 
                        idx[i]  # User ID for privacy reward
                    )
                    rewards.append(reward)
                rewards = np.array(rewards)
                
                # Get value estimates from critic
                values = self.critic.predict(gen_trajs)
                
                # Compute advantages
                advantages = self.ppo_agent.compute_advantages(values, rewards)
                
                # Update policy using PPO
                ppo_loss = self.ppo_agent.update_policy(real_trajs_bc, gen_trajs, old_probs, advantages)
                
                # Update critic
                critic_loss = self.critic.train_on_batch(gen_trajs, rewards)
                
                print(f"[{epoch}/{epochs}] D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f} | PPO Loss: {ppo_loss:.4f} | Critic Loss: {critic_loss:.4f}")
            else:
                print(f"[{epoch}/{epochs}] D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")

            # Print and save the losses/params
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)
                print('Model params saved to the disk.')

    def save_checkpoint(self, epoch):
        # Save weights
        self.generator.save_weights(f"params/G_model_{epoch}.h5")
        self.discriminator.save_weights(f"params/D_model_{epoch}.h5")
        self.combined.save_weights(f"params/C_model_{epoch}.h5")
        self.critic.save_weights(f"params/Critic_model_{epoch}.h5")