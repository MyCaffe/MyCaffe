from torch import nn
from constants import *

import sys
import torch
import math
import numpy as np
import zipfile

from test_base import DebugFunction
from layers import LinearEx
from layers import SoftmaxEx
from layers import LayerNormEx

class TestCausalSelfAttention:
    def __init__(self, vocab_size, n_embd, n_head, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        self.attn = CausalSelfAttentionEx("blk0" + ".attn", n_embd, n_head, block_size, attn_pdrop, resid_pdrop).to(device)
        self.ln_f = LayerNormEx("ln_f", n_embd).to(device)
        self.lm_head = LinearEx("lm_head", n_embd, vocab_size, bias=False).to(device)
        self.softmax = nn.LogSoftmax(dim=-1).to(device)
        self.criterion = nn.NLLLoss().to(device)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
    
    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask
        
    def test(self):
        debug = DebugFunction.apply
        src_input = torch.from_numpy(np.load('test/iter_0/1_x.npy'))
        trg_output = torch.from_numpy(np.load('test/iter_0/1_targets.npy'))
        e_mask, d_mask = self.make_mask(src_input, src_input)

        # Create input tensor
        x = torch.from_numpy(np.load('test/iter_0/1_x_emb.npy')).to(device)
        x.requires_grad = True

        DebugFunction.trace(x, "1_x_in")
        x = debug(x)
        
        # Act
        output = self.attn(x, d_mask)

        DebugFunction.trace(output, "12_out1")
        output = debug(output)
        
        # Assert
        assert output.shape == (batch_size, seq_len, self.n_embd)
        out1 = self.lm_head(output)

        DebugFunction.trace(out1, "13_out1")
        out1 = debug(out1)
        
        out2 = self.softmax(out1)

        DebugFunction.trace(out2, "14_out2")
        out2 = debug(out2)
        
        loss = self.criterion(out2.view(-1, self.vocab_size), trg_output.view(-1).to(device))

        DebugFunction.trace(loss, "15_loss")
        loss = debug(loss)

        loss.backward()
        print('done!')

class CausalSelfAttentionEx(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, tag, n_embd, n_head, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        self.tag = tag
        assert n_embd % n_head == 0
        self.softmax = SoftmaxEx(tag + ".smx", dim=-1)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = LinearEx(tag + ".c_attn", n_embd, 3 * n_embd)
        # output projection
        self.c_proj = LinearEx(tag + ".c_proj", n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_head
        self.n_embd = n_embd
        DebugFunction.set_output_path(0)
        self.load_internal_state()
        self.tag = "csa_test." + self.tag
        
    def load_internal_state(self):
        self.c_attn.set_parameters(DebugFunction.load(self.tag + ".c_attn.weight"), DebugFunction.load(self.tag + ".c_attn.bias"))
        self.c_proj.set_parameters(DebugFunction.load(self.tag + ".c_proj.weight"), DebugFunction.load(self.tag + ".c_proj.bias"))

    def forward(self, x, mask):
        debug = DebugFunction.apply
        
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        DebugFunction.trace(x, self.tag + ".1_x")
        x = debug(x)

        DebugFunction.trace(self.c_attn.weight, self.tag + ".1_c_attn.weight");
        DebugFunction.trace(self.c_attn.bias, self.tag + ".1_c_attn.bias");

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        x1 = self.c_attn(x)
        DebugFunction.trace(x1, self.tag + ".2_x1")
        x1 = debug(x1)        
        
        q, k ,v  = x1.split(self.n_embd, dim=2)
        
        DebugFunction.trace(q, self.tag + ".2_q")
        q = debug(q)
        DebugFunction.trace(k, self.tag + ".2_k")
        k = debug(k)
        DebugFunction.trace(v, self.tag + ".2_v")
        v = debug(v)
        
        kt = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        qt= q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        vt = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        DebugFunction.trace(qt, self.tag + ".3_qt")
        qt = debug(qt)
        DebugFunction.trace(kt, self.tag + ".3_kt")
        kt = debug(kt)
        DebugFunction.trace(vt, self.tag + ".3_vt")
        vt = debug(vt)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        kt1 = kt.transpose(-2, -1)
        
        DebugFunction.trace(kt1, self.tag + ".4_kt1")
        kt1 = debug(kt1)
        
        att1 = (qt @ kt1)
        
        DebugFunction.trace(att1, self.tag + ".5_att1")
        att1 = debug(att1)

        att = att1 * (1.0 / math.sqrt(k.size(-1)))
        
        DebugFunction.trace(att, self.tag + ".6_att")
        att = debug(att)
        
        attm = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        DebugFunction.trace(attm, self.tag + ".7_attm")
        attm = debug(attm)

        atts = self.softmax(attm) 

        DebugFunction.trace(atts, self.tag + ".8_atts")
        atts = debug(atts)
        
        #atts = self.attn_dropout(atts)
        
        yt = atts @ vt # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        DebugFunction.trace(yt, self.tag + ".9_yt")
        yt = debug(yt)
        
        y = yt.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        DebugFunction.trace(y, self.tag + ".10_y")
        y = debug(y)
        
        # output projection
        y = self.c_proj(y)

        DebugFunction.trace(y, self.tag + ".11_y")
        y = debug(y)
        
        #y = self.resid_dropout(y)
        return y

vocab_size = 65
n_embd = 192
n_head = 6
block_size = seq_len
attn_pdrop = 0
resid_pdrop = 0
    
DebugFunction.set_seed(1701)    
test = TestCausalSelfAttention(vocab_size, n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
test.test()

if os.path.isfile("test/_causalselfattn_test.zip"):
    os.remove('test/_causalselfattn_test.zip')

with zipfile.ZipFile('test/_causalselfattn_test.zip', 'w') as myzip:
    for file in os.listdir('test'):
        if file.endswith('.npy'):
            myzip.write(os.path.join('test', file))

print("Done")
