from torch import nn
from constants import *

import torch
import math
import numpy as np
import zipfile
from layers_ex import PositionalEncoderEx

from custom_data import set_seed
from test_base import DebugFunction

subPath = "posenc"

class TestPositionalEncoder:
    def __init__(self):
        super().__init__()
        self.vocab_size = sp_vocab_size
        self.output_linear = nn.Linear(d_model, self.vocab_size).to(device)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def test(self):
        DebugFunction.set_output_path(subPath, 0)
        trg_output = torch.from_numpy(np.load('test/trg_output.npy')).to(device)
        debug = DebugFunction.apply

        # Create encoder layer
        posenc = PositionalEncoderEx("pos").to(device)

        # Create input/output tensor
        x0 = torch.from_numpy(np.load('test/q0.npy')).to(device)
        x0.requires_grad = True
        
        # Forward
        output = posenc(0, x0)        
        DebugFunction.trace(output, "pos.output")
        output = debug(output)
        
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
test = TestPositionalEncoder()
test.test()

#if os.path.isfile("test/_posenc_test.zip"):
#    os.remove('test/_posenc_test.zip')
#
#with zipfile.ZipFile('test/_posenc_test.zip', 'w') as myzip:
#    for file in os.listdir('test'):
#        if file.endswith('.npy'):
#            myzip.write(os.path.join('test', file))
#
#print("Done")
    