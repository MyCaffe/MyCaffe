from torch import nn
from constants import *

import sys
import torch
import math
import numpy as np
import zipfile

from custom_data import set_seed
from test_base import DebugFunction
from layers import LinearEx
from layers import SoftmaxEx
from layers_ex import MultiheadAttentionEx

subPath = "mha"

class TestMultiheadAttention:
    def __init__(self):
        super().__init__()
        self.vocab_size = sp_vocab_size
        self.output_linear = nn.Linear(d_model, self.vocab_size).to(device)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
    
    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask

    def test(self):
        DebugFunction.set_output_path(subPath, 0)
        debug = DebugFunction.apply
        src_input = torch.from_numpy(np.load('test/src_input.npy')).to(device)
        trg_input = torch.from_numpy(np.load('test/trg_input.npy')).to(device)
        trg_output = torch.from_numpy(np.load('test/trg_output.npy')).to(device)
        e_mask, d_mask = self.make_mask(src_input, trg_input)

        # Create multihead attention layer
        multihead_attention = MultiheadAttentionEx("mh").to(device)
        multihead_attention.save_internal_state(0);
        
        # Create input tensor
        q = torch.from_numpy(np.load('test/q0.npy')).to(device)
        q.requires_grad = True
        k = torch.from_numpy(np.load('test/k0.npy')).to(device)
        k.requires_grad = True
        v = torch.from_numpy(np.load('test/v0.npy')).to(device)
        v.requires_grad = True

        # Act
        output = multihead_attention(0, q, k, v, e_mask)
        np.save('test/multihead_output.npy', output.detach().cpu().numpy())

        # Assert
        assert output.shape == (batch_size, seq_len, d_model)

        out1 = self.output_linear(output)

        DebugFunction.trace(out1, "13_out1")
        out1 = debug(out1)
        
        out2 = self.softmax(out1)

        DebugFunction.trace(out2, "14_out2")
        out2 = debug(out2)
        
        loss = self.criterion(out2.view(-1, self.vocab_size), trg_output.view(-1))

        DebugFunction.trace(loss, "15_loss")
        loss = debug(loss)

        loss.backward()
        print('done!')
                       
set_seed(1701)    
test = TestMultiheadAttention()
test.test()

#if os.path.isfile("test/_multihead_test.zip"):
#    os.remove('test/_multihead_test.zip')

#with zipfile.ZipFile('test/_multihead_test.zip', 'w') as myzip:
#    for file in os.listdir('test'):
#        if file.endswith('.npy'):
#            myzip.write(os.path.join('test', file))

print("Done")
