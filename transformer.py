import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        
    def forward(self, x):
        # Reshape for MultiheadAttention
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, seq_len]
        
        # Self-attention
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        x = self.layernorm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = self.layernorm2(x + ffn_output)
        
        # Reshape back
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len] -> [batch_size, seq_len, embed_dim]
        return x 