import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from PositionEmbedding import PositionEmbeddingFixedWeights
from numpy import random
import numpy as np

'''
The TransformerEncoder layer is a PyTorch rewrite of the TensorFlow and Keras version by Stefania Cristina
@see [Implementing the Transformer Encoder from Scratch](https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/) by Stefania Cristina, 2022
@see [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
'''

# AddNormalization layer adds and normalizes
class AddNormalization(nn.Module):
    def __init__(self, d_model):
        super(AddNormalization, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_x):
        # Add the sublayer output to the input and normalize
        return self.layernorm(x + sublayer_x)

# Feed Forward Layer mlp
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.proj = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Pass the input through the first linear layer with ReLU activation
        x = self.fc1(x)
        x = self.activation(x)
        # Pass the output through the second linear layer
        return self.proj(x)

# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_k, d_v, d_model, d_ff, rate):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads, d_k, d_v, d_model) # Multi-Head Attention
        self.dropout1 = nn.Dropout(rate)
        self.addnorm1 = AddNormalization(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(rate)
        self.addnorm2 = AddNormalization(d_model)

    def forward(self, x, mask):
        # Pass the input through the Multi-Head Attention layer
        mha_output = self.mha(x, x, x, mask)
        mha_output = self.dropout1(mha_output)
        
        # Pass the output through the Add and Norm layer
        add_norm1 = self.addnorm1(x, mha_output)
        
        # Pass the output through the Feed Forward layer
        ffn_output = self.ffn(add_norm1)
        ffn_output = self.dropout2(ffn_output)
        
        # Pass the output through the Add and Norm layer
        return self.addnorm2(add_norm1, ffn_output)

# Transformer Encoder 
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, sequence_length, num_heads, d_k, d_v, d_model, d_ff, num_layers, rate, batch_size):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.pos_emb = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model, batch_size) # Position Embedding
        self.dropout = nn.Dropout(rate)
        self.enc_layers = nn.ModuleList([EncoderLayer(num_heads, d_k, d_v, d_model, d_ff, rate) for _ in range(num_layers)]) # Encoder layers
        
    def to(self, device):
        new_self = super(TransformerEncoder, self).to(device)

        for enc in new_self.enc_layers:
            enc.to(device)
        
        new_self.pos_emb = new_self.pos_emb.to(device)
        return new_self

    def forward(self, x, mask):
        # Pass the input through the Position encoding layer
        x = self.pos_emb(x)
        x = self.dropout(x)
        
        # Pass the output through the Encoder Layers
        for enc in self.enc_layers:
            x = enc(x, mask)
        
        return x

    @staticmethod
    def test():
        h = 8 # Number of self-attention heads
        d_k = 64 # Dimension of the queries and keys
        d_v = 64 # Dimension of the values
        d_ff = 2048 # Demension of the fully connected mpl
        d_model = 512 # Dimension of the model
        n = 6 # Number of encoder layers
        batch_size = 64 # Batch size
        dropout_rate = 0.1 # Frequency of dropout
        enc_vocab_size = 20 # Size of the encoder vocabulary
        input_seq_length = 5 # Length of the input sequence
        
        input_seq = random.random((batch_size, input_seq_length))
        input_seq = torch.as_tensor(input_seq, dtype=torch.long)

        encoder = TransformerEncoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
        output = encoder(input_seq, None)
        
        print(output.shape)
        print(output)
        
# TransformerEncoder.test()