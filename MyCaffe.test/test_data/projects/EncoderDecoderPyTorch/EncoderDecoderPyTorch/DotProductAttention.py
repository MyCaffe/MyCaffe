import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from numpy import random

'''
The DotProductAttention layer is a PyTorch rewrite of the TensorFlow and Keras version by Stefania Cristina
@see [How to Implement Multi-Head Attention from Scratch in TensorFlow and Keras](https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/) by Stefania Cristina, 2022
@see [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
'''

# Scaled Dot-Product Attention layer
class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
      
    def forward(self, Q, K, V, mask=None):
        # Q: [batch_size, num_heads, seq_len_q, d_k]
        # K: [batch_size, num_heads, seq_len_k, d_k]
        # V: [batch_size, num_heads, seq_len_v, d_v]
        # mask: [batch_size, num_heads, seq_len_q, seq_len_k]
        # output: [batch_size, num_heads, seq_len_q, d_v]
        # attention: [batch_size, num_heads, seq_len_q, seq_len_k]
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            mask1 = mask.view(mask.shape[0], 1, 1, mask.shape[-1])
            scores += (mask1 * -1e9)
        
        attention = F.softmax(scores, dim=-1)

        return torch.matmul(attention, V)

    
    @staticmethod
    def test():
        d_k = 4 # Dimensionality of the linearly projected queries and keys
        d_v = 4 # Dimensionality of the linearly projected values
        batch_size = 2 # Batch size from the training process        
        block_size = 5 # Maximum length of the input sequence.
        
        random.seed(1703)
        
        queries = torch.from_numpy(random.random((batch_size, block_size, d_k)))
        keys = torch.from_numpy(random.random((batch_size, block_size, d_k)))
        values = torch.from_numpy(random.random((batch_size, block_size, d_v)))
        
        attention = DotProductAttention();
        context = attention(queries, keys, values)

        print("Context ", context.shape, "\n")
        print(context)
    
# DotProductAttention.test()        