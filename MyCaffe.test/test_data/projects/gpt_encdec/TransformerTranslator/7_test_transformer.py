from torch import nn
from constants import *
from layers import LinearEx
from layers import LogSoftmaxEx
from custom_data import set_seed

import torch
import math
import numpy as np
import zipfile

from test_base import DebugFunction
from layers_ex import PositionalEncoderEx
from layers_ex import EncoderLayerEx
from layers_ex import DecoderLayerEx
from layers_ex import LayerNormalizationEx
from layers_ex import TransformerEx

subPath = "transformer"

class TestTransformer:
    def __init__(self):
        super().__init__()
        self.vocab_size = 16000
        
    def test(self):
        DebugFunction.set_output_path(subPath, 0)
        debug = DebugFunction.apply
        src_input = torch.from_numpy(np.load('test/src_input.npy')).to(device)
        trg_input = torch.from_numpy(np.load('test/trg_input.npy')).to(device)
        trg_output = torch.from_numpy(np.load('test/trg_output.npy')).to(device)
        e_mask, d_mask = TestTransformer.make_mask(src_input, trg_input)
        
        np.save('test/e_mask.npy', e_mask.detach().cpu().numpy())
        np.save('test/d_mask.npy', d_mask.detach().cpu().numpy())

        # Create the transformer
        model = TransformerEx(src_vocab_size=self.vocab_size, trg_vocab_size=self.vocab_size).to(device)
        criterion = nn.NLLLoss()

        model.save_internal_state(0)

        # Forward
        out2 = model(0, src_input, trg_input, e_mask, d_mask)
        
        DebugFunction.trace(out2, "14_out2")
        out2 = debug(out2)
        
        loss = criterion(out2.view(-1, self.vocab_size),
                         trg_output.view(trg_output.shape[0] * trg_output.shape[1]))        

        sumout = out2.sum()
        sumout /= (3 * 200 * 16000)

        DebugFunction.trace(loss, "15_loss")
        loss = debug(loss)
        
        loss.backward()
        print('done!')

    @staticmethod
    def make_mask(src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask
    
set_seed(1701)    
test = TestTransformer()
test.test()

#if os.path.isfile("test/_transformer_test.zip"):
#    os.remove('test/_transformer_test.zip')

#with zipfile.ZipFile('test/_transformer_test.zip', 'w') as myzip:
#    for file in os.listdir('test'):
#        if file.endswith('.npy'):
#            myzip.write(os.path.join('test', file))

print("Done")
    
