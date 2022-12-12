import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from numpy import random

from DotProductAttention import DotProductAttention

'''
The MultiHeadAttention layer is a PyTorch rewrite of the TensorFlow and Keras version by Stefania Cristina
@see [How to Implement Multi-Head Attention from Scratch in TensorFlow and Keras](https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/) by Stefania Cristina, 2022
@see [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
'''

# Multi-Head Attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_k, d_v, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, d_k, bias=True)
        self.w_ks = nn.Linear(d_model, d_k, bias=True)
        self.w_vs = nn.Linear(d_model, d_v, bias=True)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.constant_(self.w_qs.bias, 0)
        nn.init.constant_(self.w_ks.bias, 0)
        nn.init.constant_(self.w_vs.bias, 0)
        
        self.attention = DotProductAttention()
        self.fc = nn.Linear(d_v, d_model, bias=True)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: [batch_size, num_heads, seq_len, -1]
            x = x.view(x.shape[0], x.shape[1], heads, -1)
            x = x.transpose(1,2)
        else:
            # Tensor shape after reshaping and transposing: [batch_size, seq_len, d_k]
            x = x.transpose(1,2)
            x = x.contiguous().view(x.shape[0], x.shape[1], self.d_k)
        return x
        
    def forward(self, Q, K, V, mask=None):
        # Q: [batch_size, seq_len_q, d_model]
        # K: [batch_size, seq_len_k, d_model]
        # V: [batch_size, seq_len_v, d_model]
        # mask: [batch_size, seq_len_q, seq_len_k]
        # output: [batch_size, seq_len_q, d_model]
        # attention: [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # Rearrange Q so that all heads can be computed in parallel.
        q1 = self.w_qs(Q)
        q_reshaped = self.reshape_tensor(q1, self.num_heads, True)
        # Rearrange K so that all heads can be computed in parallel.
        k_reshaped = self.reshape_tensor(self.w_ks(K), self.num_heads, True)
        # Rearrange V so that all heads can be computed in parallel.
        v_reshaped = self.reshape_tensor(self.w_vs(V), self.num_heads, True)

        # Compute the multi-head attention output using reshaped Q, K, and V
        o_reshaped = self.attention.forward(q_reshaped, k_reshaped, v_reshaped, mask)

        # Rearrange back to concatenated form: (batch_size, seq_len, d_v)
        output = self.reshape_tensor(o_reshaped, self.num_heads, False)

        # Apply final linear projection
        return self.fc(output)
        
    @staticmethod
    def test():
        h = 2 # Number of heads
        d_k = 4 # Dimensionality of the linearly projected queries and keys
        d_v = 4 # Dimensionality of the linearly projected values
        d_model = 24 # Dimensionality of the linear projection
        batch_size = 2 # Batch size from the training process        
        block_size = 5 # Maximum length of the input sequence.
        
        random.seed(1703)
        
        Q = torch.from_numpy(random.random((batch_size, block_size, d_k))).to(torch.float32)
        K = torch.from_numpy(random.random((batch_size, block_size, d_k))).to(torch.float32)
        V = torch.from_numpy(random.random((batch_size, block_size, d_v))).to(torch.float32)
        
        multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        output = multihead_attention(Q, K, V)

        print(output.shape)
        print(output)

# MultiHeadAttention.test()
