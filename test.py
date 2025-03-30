import sys
from MARC.core.logger import Logger

logger = Logger()

if len(sys.argv) < 3:
    print('Please run as:')
    print('\tpython test.py', 'TRAIN_FILE', 'TEST_FILE', 'EMBEDDING_SIZE')
    exit()

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
EMBEDDER_SIZE = int(sys.argv[3])

print('====================================', 'PARAMS',
      '====================================')
print('TRAIN_FILE =', TRAIN_FILE)
print('TEST_FILE =', TEST_FILE)
print('EMBEDDER_SIZE =', EMBEDDER_SIZE)

###############################################################################
#   LOAD DATA
###############################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from MARC.core.utils.geohash import bin_geohash

def recreate_model_structure():
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Embedding, Concatenate
    from tensorflow.keras.initializers import he_uniform
    from tensorflow.keras.regularizers import l1
    
    # Input layers
    input_day = Input(shape=(144,), name='input_day')
    input_hour = Input(shape=(144,), name='input_hour')
    input_category = Input(shape=(144,), name='input_category')
    input_lat_lon = Input(shape=(144, 40), name='input_lat_lon')
    
    # Embedding layers
    emb_day = Embedding(7, 100, input_length=144, name='emb_day')(input_day)
    emb_hour = Embedding(24, 100, input_length=144, name='emb_hour')(input_hour)
    emb_category = Embedding(10, 100, input_length=144, name='emb_category')(input_category)
    emb_lat_lon = Dense(units=100, kernel_initializer=he_uniform(seed=1), name='emb_lat_lon')(input_lat_lon)
    
    # Concatenate embeddings
    concat = Concatenate(axis=2)([emb_day, emb_hour, emb_category, emb_lat_lon])
    
    # Dropout and LSTM
    dropout1 = Dropout(0.5)(concat)
    lstm = LSTM(units=50, recurrent_regularizer=l1(0.02))(dropout1)
    
    # Final dropout and output layer
    dropout2 = Dropout(0.5)(lstm)
    output = Dense(units=193, kernel_initializer=he_uniform(), activation='softmax')(dropout2)
    
    # Create model
    model = Model(inputs=[input_day, input_hour, input_category, input_lat_lon], 
                  outputs=output)
    
    return model
    
def get_trajectories(train_file, test_file, tid_col='tid',
                     label_col='label', geo_precision=8, drop=[]):
    file_str = "'" + train_file + "' and '" + test_file + "'"
    logger.log(Logger.INFO, "Loading data from file(s) " + file_str + "... ")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    df = pd.concat([df_train.copy(), df_test], ignore_index=True)
    tids_train = df_train[tid_col].unique()

    keys = list(df.keys())
    vocab_size = {}
    keys.remove(tid_col)

    for col in drop:
        if col in keys:
            keys.remove(col)
            logger.log(Logger.WARNING, "Column '" + col + "' dropped " +
                       "from input file!")
        else:
            logger.log(Logger.WARNING, "Column '" + col + "' cannot be " +
                       "dropped because it was not found!")

    num_classes = len(set(df[label_col]))
    count_attr = 0
    lat_lon = False

    if 'lat' in keys and 'lon' in keys:
        keys.remove('lat')
        keys.remove('lon')
        lat_lon = True
        count_attr += geo_precision * 5
        logger.log(Logger.INFO, "Attribute Lat/Lon: " +
                   str(geo_precision * 5) + "-bits value")

    for attr in keys:
        df[attr] = LabelEncoder().fit_transform(df[attr])
        vocab_size[attr] = max(df[attr]) + 1

        if attr != label_col:
            values = len(set(df[attr]))
            count_attr += values
            logger.log(Logger.INFO, "Attribute '" + attr + "': " +
                       str(values) + " unique values")

    logger.log(Logger.INFO, "Total of attribute/value pairs: " +
               str(count_attr))
    keys.remove(label_col)

    x = [[] for key in keys]
    y = []
    idx_train = []
    idx_test = []
    max_length = 0
    trajs = len(set(df[tid_col]))

    if lat_lon:
        x.append([])

    for idx, tid in enumerate(set(df[tid_col])):
        logger.log_dyn(Logger.INFO, "Processing trajectory " + str(idx + 1) +
                       "/" + str(trajs) + ". ")
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, keys].values)

        for i in range(0, len(features)):
            x[i].append(features[i])

        if lat_lon:
            loc_list = []
            for i in range(0, len(traj)):
                lat = traj['lat'].values[i]
                lon = traj['lon'].values[i]
                loc_list.append(bin_geohash(lat, lon, geo_precision))
            x[-1].append(loc_list)

        label = traj[label_col].iloc[0]
        y.append(label)

        if tid in tids_train:
            idx_train.append(idx)
        else:
            idx_test.append(idx)

        if traj.shape[0] > max_length:
            max_length = traj.shape[0]

    if lat_lon:
        keys.append('lat_lon')
        vocab_size['lat_lon'] = geo_precision * 5

    one_hot_y = OneHotEncoder().fit(df.loc[:, [label_col]])

    
    # Replace the x_padded section with this more robust approach
    x_padded = []
    for feature_list in x:
        # Each feature might have a complex structure
        padded_feature = []
        for traj_idx in range(len(feature_list)):
            # If this is a simple sequence, we can directly use it
            # If it's a complex structure, we need special handling
            try:
                # Try to convert to a regular array first
                padded_feature.append(np.array(feature_list[traj_idx]))
            except ValueError:
                # If that fails, we need to handle the complex structure
                # This depends on your specific data structure
                # For this example, let's assume we can convert to a string representation
                padded_feature.append(str(feature_list[traj_idx]))
        
        # Convert the entire feature list to an object array which can hold different types
        x_padded.append(np.array(padded_feature, dtype=object))
    
    x = x_padded
    y = one_hot_y.transform(pd.DataFrame(y)).toarray()
    logger.log(Logger.INFO, "Loading data from files " + file_str + "... DONE!")
    
    x_train = np.asarray([f[idx_train] for f in x])
    y_train = y[idx_train]
    x_test = np.asarray([f[idx_test] for f in x])
    y_test = y[idx_test]

    logger.log(Logger.INFO, 'Trajectories:  ' + str(trajs))
    logger.log(Logger.INFO, 'Labels:        ' + str(len(y[0])))
    logger.log(Logger.INFO, 'Train size:    ' + str(len(x_train[0]) / trajs))
    logger.log(Logger.INFO, 'Test size:     ' + str(len(x_test[0]) / trajs))
    logger.log(Logger.INFO, 'x_train shape: ' + str(x_train.shape))
    logger.log(Logger.INFO, 'y_train shape: ' + str(y_train.shape))
    logger.log(Logger.INFO, 'x_test shape:  ' + str(x_test.shape))
    logger.log(Logger.INFO, 'y_test shape:  ' + str(y_test.shape))

    return (keys, vocab_size, num_classes, max_length,
            x_train, x_test,
            y_train, y_test)


(keys, vocab_size,
 num_classes,
 max_length,
 x_train, x_test,
 y_train, y_test) = get_trajectories(train_file=TRAIN_FILE,
                                     test_file=TEST_FILE,
                                     tid_col='tid',
                                     label_col='label')


###############################################################################
#   PREPARING CLASSIFIER DATA
###############################################################################
from keras.preprocessing.sequence import pad_sequences


cls_x_train = [pad_sequences(f, max_length, padding='pre') for f in x_train]
cls_x_test = [pad_sequences(f, max_length, padding='pre') for f in x_test]
cls_y_train = y_train
cls_y_test = y_test


###############################################################################
#   CLASSIFIER
###############################################################################
from keras.models import Model
from keras.layers import Dense, LSTM, GRU, Dropout
from keras.initializers import he_uniform
from keras.regularizers import l1
from keras.optimizers import Adam
from keras.layers import Input, Add, Average, Concatenate, Embedding
from keras.callbacks import EarlyStopping
from MARC.core.utils.metrics import compute_acc_acc5_f1_prec_rec


CLASS_DROPOUT = 0.5
CLASS_HIDDEN_UNITS = 50
CLASS_LRATE = 0.001
CLASS_BATCH_SIZE = 64
CLASS_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 30
BASELINE_METRIC = 'acc'
BASELINE_VALUE = 0.5


print('=====================================', 'OURS',
      '=====================================')

print('CLASS_DROPOUT =', CLASS_DROPOUT)
print('CLASS_HIDDEN_UNITS =', CLASS_HIDDEN_UNITS)
print('CLASS_LRATE =', CLASS_LRATE)
print('CLASS_BATCH_SIZE =', CLASS_BATCH_SIZE)
print('CLASS_EPOCHS =', CLASS_EPOCHS)
print('EARLY_STOPPING_PATIENCE =', EARLY_STOPPING_PATIENCE)
print('BASELINE_METRIC =', BASELINE_METRIC)
print('BASELINE_VALUE =', BASELINE_VALUE, '\n')


class EpochLogger(EarlyStopping):

    def __init__(self, metric='val_acc', baseline=0):
        super(EpochLogger, self).__init__(monitor='val_acc',
                                          mode='max',
                                          patience=EARLY_STOPPING_PATIENCE)
        self._metric = metric
        self._baseline = baseline
        self._baseline_met = False

    def on_epoch_begin(self, epoch, logs={}):
        print("===== Training Epoch %d =====" % (epoch + 1))

        if self._baseline_met:
            super(EpochLogger, self).on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        pred_y_train = np.array(self.model.predict(cls_x_train))
        (train_acc,
         train_acc5,
         train_f1_macro,
         train_prec_macro,
         train_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_train,
                                                         pred_y_train,
                                                         print_metrics=True,
                                                         print_pfx='TRAIN')
        
        # Add utility metrics for training data
        train_spatial, train_temporal, train_category, train_overall = compute_utility_metrics(
            cls_y_train, 
            pred_y_train,
            lat_lon_data=x_train[-1] if 'lat_lon' in keys else None,
            time_data=[x_train[keys.index('day')], x_train[keys.index('hour')]] 
                if 'day' in keys and 'hour' in keys else [None, None],
            category_data=x_train[keys.index('category')] if 'category' in keys else None,
            print_metrics=True,
            print_pfx='TRAIN'
        )

        pred_y_test = np.array(self.model.predict(cls_x_test))
        (test_acc,
         test_acc5,
         test_f1_macro,
         test_prec_macro,
         test_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_test,
                                                        pred_y_test,
                                                        print_metrics=True,
                                                        print_pfx='TEST')
        
        # Add utility metrics for test data
        test_spatial, test_temporal, test_category, test_overall = compute_utility_metrics(
            cls_y_test, 
            pred_y_test,
            lat_lon_data=x_test[-1] if 'lat_lon' in keys else None,
            time_data=[x_test[keys.index('day')], x_test[keys.index('hour')]] 
                if 'day' in keys and 'hour' in keys else [None, None],
            category_data=x_test[keys.index('category')] if 'category' in keys else None,
            print_metrics=True,
            print_pfx='TEST'
        )
        
        
        print(test_acc)
        print(test_acc5)
        print(test_f1_macro)
        print(test_prec_macro)
        print(test_rec_macro)
        print(test_spatial)
        print(test_temporal)
        print(test_category)
        print(test_overall)

    def on_train_begin(self, logs=None):
        super(EpochLogger, self).on_train_begin(logs)

    def on_train_end(self, logs=None):
        if self._baseline_met:
            super(EpochLogger, self).on_train_end(logs)

inputs = []
embeddings = []

for idx, key in enumerate(keys):
    if key == 'lat_lon':
        i = Input(shape=(max_length, vocab_size[key]),
                  name='input_' + key)
        e = Dense(units=EMBEDDER_SIZE,
                  kernel_initializer=he_uniform(seed=1),
                  name='emb_' + key)(i)
    else:
        i = Input(shape=(max_length,),
                  name='input_' + key)
        e = Embedding(vocab_size[key],
                      EMBEDDER_SIZE,
                      input_length=max_length,
                      name='emb_' + key)(i)
    inputs.append(i)
    embeddings.append(e)

if len(embeddings) == 1:
    hidden_input = embeddings[0]
else:
    hidden_input = Concatenate(axis=2)(embeddings)

hidden_dropout = Dropout(CLASS_DROPOUT)(hidden_input)

rnn_cell = LSTM(units=CLASS_HIDDEN_UNITS, recurrent_regularizer=l1(0.02))(hidden_dropout)

rnn_dropout = Dropout(CLASS_DROPOUT)(rnn_cell)

softmax = Dense(units=num_classes,
                kernel_initializer=he_uniform(),
                activation='softmax')(rnn_dropout)

classifier = Model(inputs=inputs, outputs=softmax)
opt = Adam(learning_rate=CLASS_LRATE)

classifier.compile(optimizer=opt,
                   loss='categorical_crossentropy',
                   metrics=['acc', 'top_k_categorical_accuracy'])

from keras.models import load_model
from keras.models import model_from_json

# load json and create model
json_file = open('MARC/MARC.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

classifier2 = recreate_model_structure()  # You'd need to define this function
classifier2.load_weights('MARC/MARC_Weight.h5')

pred_y_test = np.array(classifier2.predict(cls_x_test))
(test_acc,
 test_acc5,
 test_f1_macro,
 test_prec_macro,
 test_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_test,
                                                pred_y_test,
                                                print_metrics=True,
                                                print_pfx='TEST')
print(test_acc)
print(test_acc5)
print(test_f1_macro)
print(test_prec_macro)
print(test_rec_macro)

def compute_utility_metrics(y_true, y_pred, lat_lon_data, time_data, category_data, print_metrics=True, print_pfx=""):
    """
    Compute utility metrics for spatial, temporal, and category dimensions
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        lat_lon_data: Spatial data
        time_data: Temporal data (day and hour)
        category_data: Category data
        print_metrics: Whether to print metrics
        print_pfx: Prefix for printing
        
    Returns:
        spatial_score: Spatial utility score
        temporal_score: Temporal utility score
        category_score: Category utility score
        overall_utility: Overall combined utility score
    """
    from sklearn.metrics import pairwise_distances
    import numpy as np
    
    # Get predicted classes
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Spatial utility metric
    spatial_distances = []
    
    # First check if lat_lon_data is available
    if lat_lon_data is not None:
        # Create a safer function to compute centroid
        def compute_centroid(indices):
            try:
                # Try to stack the arrays first to ensure they're compatible
                data_points = []
                for idx in indices:
                    if idx < len(lat_lon_data):
                        # Convert to numpy array and flatten if needed
                        point = np.array(lat_lon_data[idx])
                        if len(point.shape) > 1:
                            # If it's a sequence of points, take the mean
                            point = np.mean(point, axis=0)
                        data_points.append(point)
                
                if data_points:
                    # Convert list to array and compute mean
                    return np.mean(np.array(data_points), axis=0)
                return None
            except:
                return None
        
        for i, (true_class, pred_class) in enumerate(zip(y_true_classes, y_pred_classes)):
            if true_class == pred_class:
                spatial_distances.append(1.0)  # Perfect match
            else:
                # Calculate distance between lat/lon centroids of classes
                true_indices = np.where(y_true_classes == true_class)[0]
                pred_indices = np.where(y_pred_classes == pred_class)[0]
                
                true_centroid = compute_centroid(true_indices)
                pred_centroid = compute_centroid(pred_indices)
                
                if true_centroid is not None and pred_centroid is not None:
                    # Calculate geo-distance (simplified Euclidean for this example)
                    dist = np.sqrt(np.sum((true_centroid - pred_centroid)**2))
                    
                    # Compute max distance for normalization
                    class_centroids = []
                    for cls in np.unique(y_true_classes):
                        cls_indices = np.where(y_true_classes == cls)[0]
                        cls_centroid = compute_centroid(cls_indices)
                        if cls_centroid is not None:
                            class_centroids.append(cls_centroid)
                    
                    if len(class_centroids) >= 2:
                        class_centroids = np.array(class_centroids)
                        max_dist = np.max(pairwise_distances(class_centroids))
                        spatial_distances.append(1.0 - min(dist/max_dist, 1.0))
                    else:
                        spatial_distances.append(0.0)
                else:
                    spatial_distances.append(0.0)
        
        spatial_score = np.mean(spatial_distances) if spatial_distances else 0.0
    else:
        # If lat_lon_data is not available, use accuracy as spatial score
        spatial_score = np.mean(y_pred_classes == y_true_classes)
    
    # Temporal utility metric
    # Measuring how temporally close predictions are to ground truth
    day_distances = []
    hour_distances = []
    
    if time_data[0] is not None and time_data[1] is not None:
        # Create a safer function to compute temporal centroid
        def compute_temporal_centroid(indices, data):
            try:
                values = []
                for idx in indices:
                    if idx < len(data):
                        # Handle the case when data[idx] is a sequence
                        point = np.array(data[idx])
                        if len(point.shape) > 0:  # If it's a sequence
                            point = np.mean(point)
                        values.append(point)
                
                if values:
                    return np.mean(values)
                return None
            except:
                return None
        
        for i, (true_class, pred_class) in enumerate(zip(y_true_classes, y_pred_classes)):
            if true_class == pred_class:
                day_distances.append(1.0)
                hour_distances.append(1.0)
            else:
                # Calculate temporal distance for day of week (cyclic)
                true_day_indices = np.where(y_true_classes == true_class)[0]
                pred_day_indices = np.where(y_pred_classes == pred_class)[0]
                
                true_day_centroid = compute_temporal_centroid(true_day_indices, time_data[0])
                pred_day_centroid = compute_temporal_centroid(pred_day_indices, time_data[0])
                
                if true_day_centroid is not None and pred_day_centroid is not None:
                    # Days are cyclic (0-6), so max distance is 3.5
                    day_diff = min(abs(true_day_centroid - pred_day_centroid), 
                                   7 - abs(true_day_centroid - pred_day_centroid))
                    day_distances.append(1.0 - min(day_diff/3.5, 1.0))
                else:
                    day_distances.append(0.0)
                
                # Calculate temporal distance for hour (cyclic)
                true_hour_centroid = compute_temporal_centroid(true_day_indices, time_data[1])
                pred_hour_centroid = compute_temporal_centroid(pred_day_indices, time_data[1])
                
                if true_hour_centroid is not None and pred_hour_centroid is not None:
                    # Hours are cyclic (0-23), so max distance is 12
                    hour_diff = min(abs(true_hour_centroid - pred_hour_centroid), 
                                    24 - abs(true_hour_centroid - pred_hour_centroid))
                    hour_distances.append(1.0 - min(hour_diff/12.0, 1.0))
                else:
                    hour_distances.append(0.0)
        
        day_score = np.mean(day_distances) if day_distances else 0.0
        hour_score = np.mean(hour_distances) if hour_distances else 0.0
        temporal_score = (day_score + hour_score) / 2
    else:
        # If temporal data is not available, use accuracy as temporal score
        day_score = hour_score = temporal_score = np.mean(y_pred_classes == y_true_classes)
    
    # Category utility metric
    category_similarities = []
    
    if category_data is not None:
        # Create a safer function to extract category values
        def extract_categories(indices):
            try:
                categories = []
                for idx in indices:
                    if idx < len(category_data):
                        # Handle the case when category_data[idx] is a sequence
                        cat = category_data[idx]
                        if hasattr(cat, '__iter__') and not isinstance(cat, (str, bytes)):
                            categories.extend([int(c) for c in cat])
                        else:
                            categories.append(int(cat))
                return categories
            except:
                return []
        
        for i, (true_class, pred_class) in enumerate(zip(y_true_classes, y_pred_classes)):
            if true_class == pred_class:
                category_similarities.append(1.0)
            else:
                # Calculate category similarity
                true_cat_indices = np.where(y_true_classes == true_class)[0]
                pred_cat_indices = np.where(y_pred_classes == pred_class)[0]
                
                true_categories = extract_categories(true_cat_indices)
                pred_categories = extract_categories(pred_cat_indices)
                
                if true_categories and pred_categories:
                    # Use vocab_size if available, otherwise infer from data
                    max_category = max(max(true_categories, default=0), max(pred_categories, default=0)) + 1
                    
                    true_cat_dist = np.bincount(true_categories, minlength=max_category)
                    true_cat_dist = true_cat_dist / (np.sum(true_cat_dist) or 1.0)
                    
                    pred_cat_dist = np.bincount(pred_categories, minlength=max_category)
                    pred_cat_dist = pred_cat_dist / (np.sum(pred_cat_dist) or 1.0)
                    
                    # Calculate cosine similarity between category distributions
                    true_norm = np.sqrt(np.dot(true_cat_dist, true_cat_dist)) or 1.0
                    pred_norm = np.sqrt(np.dot(pred_cat_dist, pred_cat_dist)) or 1.0
                    
                    similarity = np.dot(true_cat_dist, pred_cat_dist) / (true_norm * pred_norm)
                    category_similarities.append(max(similarity, 0.0))
                else:
                    category_similarities.append(0.0)
        
        category_score = np.mean(category_similarities) if category_similarities else 0.0
    else:
        # If category data is not available, use accuracy as category score
        category_score = np.mean(y_pred_classes == y_true_classes)
    
    # Overall utility score
    overall_utility = (spatial_score + temporal_score + category_score) / 3
    
    if print_metrics:
        print(f"[{print_pfx}] Spatial Utility:   {spatial_score:.4f}")
        print(f"[{print_pfx}] Temporal Utility:  {temporal_score:.4f}")
        print(f"[{print_pfx}] Day Utility:       {day_score:.4f}")
        print(f"[{print_pfx}] Hour Utility:      {hour_score:.4f}")
        print(f"[{print_pfx}] Category Utility:  {category_score:.4f}")
        print(f"[{print_pfx}] Overall Utility:   {overall_utility:.4f}")
    
    return spatial_score, temporal_score, category_score, overall_utility

pred_y_test = np.array(classifier2.predict(cls_x_test))
(test_acc,
 test_acc5,
 test_f1_macro,
 test_prec_macro,
 test_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_test,
                                                pred_y_test,
                                                print_metrics=True,
                                                print_pfx='TEST')

# Add utility metrics for the final model evaluation
test_spatial, test_temporal, test_category, test_overall = compute_utility_metrics(
    cls_y_test, 
    pred_y_test,
    lat_lon_data=x_test[-1] if 'lat_lon' in keys else None,
    time_data=[x_test[keys.index('day')], x_test[keys.index('hour')]] 
        if 'day' in keys and 'hour' in keys else [None, None],
    category_data=x_test[keys.index('category')] if 'category' in keys else None,
    print_metrics=True,
    print_pfx='TEST'
)

print(test_acc)
print(test_acc5)
print(test_f1_macro)
print(test_prec_macro)
print(test_rec_macro)
print(test_spatial)
print(test_temporal)
print(test_category)
print(test_overall)