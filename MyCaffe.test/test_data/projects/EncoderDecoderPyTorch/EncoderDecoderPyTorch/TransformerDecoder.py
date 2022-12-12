import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from PositionEmbedding import PositionEmbeddingFixedWeights
from TransformerEncoder import AddNormalization
from TransformerEncoder import FeedForward
from numpy import random
import numpy as np

'''
The TransformerDecoder layer is a PyTorch rewrite of the TensorFlow and Keras version by Stefania Cristina
@see [Implementing the Transformer Decoder from Scratch](https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/) by Stefania Cristina, 2022
@see [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
'''

# DecoderLayer class
class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_k, d_v, d_model, d_ff, rate):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(num_heads, d_k, d_v, d_model) # Self Attention
        self.dropout1 = nn.Dropout(rate)
        self.addnorm1 = AddNormalization(d_model)
        self.mha2 = MultiHeadAttention(num_heads, d_k, d_v, d_model) # Multi-Head Attention connecting Encoder to Decoder
        self.dropout2 = nn.Dropout(rate)
        self.addnorm2 = AddNormalization(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout3 = nn.Dropout(rate)
        self.addnorm3 = AddNormalization(d_model)
        
    def forward(self, x, encoder_output, lookahead_mask, mask):
        # Pass the input through the Multi-Head Attention layer
        mha_output1 = self.mha1(x, x, x, lookahead_mask)
        mha_output1 = self.dropout1(mha_output1)

        # Pass the output through the Add and Norm layer
        add_norm1 = self.addnorm1(x, mha_output1)

        # Pass the output through the Multi-Head Attention layer
        mha_output2 = self.mha2(add_norm1, encoder_output, encoder_output, mask)
        mha_output2 = self.dropout2(mha_output2)

        # Pass the output through the Add and Norm layer
        add_norm2 = self.addnorm2(add_norm1, mha_output2)
        
        # Pass the output through the Feed Forward layer
        ffn_output = self.ffn(add_norm2)
        
        # Pass the output through the Add and Norm layer
        return self.addnorm3(add_norm2, ffn_output)

# TransformerDecoder class
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, sequence_length, num_heads, d_k, d_v, d_model, d_ff, num_layers, rate, batch_size):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.pos_emb = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model, batch_size)
        self.dropout = nn.Dropout(rate)
        self.dec_layers = nn.ModuleList([DecoderLayer(num_heads, d_k, d_v, d_model, d_ff, rate) for _ in range(num_layers)])
        
    def to(self, device):
        new_self = super(TransformerDecoder, self).to(device)

        for dec in new_self.dec_layers:
            dec.to(device)
        
        new_self.pos_emb = new_self.pos_emb.to(device)
        return new_self

    def forward(self, x, encoder_output, lookahead_mask, mask):
        # Generate the positional encoding
        pos_encoding_output = self.pos_emb(x)
        
        # Add the dropout layer.
        x = self.dropout(pos_encoding_output)
        
        # Pass the output through the decoder Layers
        for dec in self.dec_layers:
            x = dec(x, encoder_output, lookahead_mask, mask)
            
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
        dec_vocab_size = 20 # Size of the decoder vocabulary
        input_seq_length = 5 # Length of the input sequence
        
        input_seq = random.random((batch_size, input_seq_length))
        input_seq = torch.as_tensor(input_seq, dtype=torch.long)
        enc_output = random.random((batch_size, input_seq_length, d_model))
        enc_output = torch.as_tensor(enc_output, dtype=torch.long)

        decoder = TransformerDecoder(dec_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
        output = decoder(input_seq, enc_output, None, None)

        print(output.shape)
        print(output)

# TransformerDecoder.test()        
        
    
    
