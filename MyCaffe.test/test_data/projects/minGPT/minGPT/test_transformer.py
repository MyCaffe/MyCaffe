from torch import nn
from constants import *
from layers import LinearEx
from layers import LogSoftmaxEx

import torch
import math
import numpy as np
import zipfile

from test_base import DebugFunction
from test_causalselfattention import CausalSelfAttentionEx
from layers import *
from constants import *

class TestTransformer:
    def __init__(self):
        super().__init__()
        self.vocab_size = 65
        
    def test(self):
        DebugFunction.set_output_path(0)
        debug = DebugFunction.apply
        x = torch.from_numpy(np.load('test/iter_0/1_x_emb.npy')).to(device)
        x.requires_grad = True
        y = torch.from_numpy(np.load('test/iter_0/1_targets.npy')).to(device)
        
        DebugFunction.trace(x, "1_x_emb1")
        x = debug(x)

        # Create the transformer
        model = BlockEx("blk0", self.vocab_size, d_model, n_head, seq_len, drop_out_rate, drop_out_rate).to(device)
        lm_head = LinearEx("lm_head", d_model, self.vocab_size, bias=False).to(device)
        softmax = LogSoftmaxEx("f_smx", dim = -1).to(device)
        criterion = nn.NLLLoss()

        model.save_internal_state()

        # Forward
        out2 = model(x)        
        DebugFunction.trace(out2, "12b_out2")
        out2 = debug(out2)
        
        logits = lm_head(out2)
        DebugFunction.trace(logits, "13_logits")
        logits = debug(logits)

        prob = softmax(logits)
        DebugFunction.trace(prob, "14_prob")
        prob = debug(prob)

        loss = criterion(prob.view(-1, self.vocab_size),
                         y.view(y.shape[0] * y.shape[1]))        

        DebugFunction.trace(loss, "15_loss")
        loss = debug(loss)
        
        loss.backward()
        print('done!')


class BlockEx(nn.Module):
    """ an unassuming Transformer block """
    
    def __init__(self, tag, vocab_size, n_embd, n_head, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        self.tag = tag
        self.ln_1 = LayerNormEx(tag + ".ln1", n_embd)
        self.attn = CausalSelfAttentionEx(tag + ".attn", n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
        self.ln_2 = LayerNormEx(tag + ".ln2", n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = LinearEx(tag + ".c_fc", n_embd, 4 * n_embd),
            c_proj  = LinearEx(tag + ".c_proj", 4 * n_embd, n_embd),
            act     = nn.ReLU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def save_internal_state(self):
        self.attn.save_internal_state()
        self.mlp["c_fc"].save_internal_state()
        self.mlp["c_proj"].save_internal_state()

    def forward(self, x):
        debug = DebugFunction.apply
        DebugFunction.trace(x, self.tag + ".1_x")
        x = debug(x)

        ln1_x = self.ln_1(x)
        DebugFunction.trace(ln1_x, self.tag + ".2_ln1_x")
        ln1_x = debug(ln1_x)

        attn_x = self.attn(ln1_x)

        DebugFunction.trace(attn_x, self.tag + ".3_attn_x")
        attn_x = debug(attn_x)

        x1 = x + attn_x

        DebugFunction.trace(x1, self.tag + ".4_x")
        x1 = debug(x1)

        ln2_x = self.ln_2(x1)

        DebugFunction.trace(ln2_x, self.tag + ".5_ln2_x")
        ln2_x = debug(ln2_x)

        mlp_x = self.mlpf(ln2_x)

        DebugFunction.trace(mlp_x, self.tag + ".6_mlp_x")
        mlp_x = debug(mlp_x)

        x = x1 + mlp_x

        DebugFunction.trace(x, self.tag + ".7_x")
        x = debug(x)

        return x
    
DebugFunction.set_seed(1701)    
test = TestTransformer()
test.test()

if os.path.isfile("test/_transformer_test.zip"):
    os.remove('test/_transformer_test.zip')

with zipfile.ZipFile('test/_transformer_test.zip', 'w') as myzip:
    for file in os.listdir('test'):
        if file.endswith('.npy'):
            myzip.write(os.path.join('test', file))

print("Done")
    

