import os
cwd = os.getcwd()
import copy
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork, GatedLinearUnit
from tft_torch.base_blocks import TimeDistributed, NullTransform
from tft_helper import apply_temporal_selection

os.chdir(cwd)
print(os.getcwd())

strSubPath = "grn"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPathWt = "data/favorita/weights/"
strPath = "test/iter_0.base_set/"

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

x = np.load(strPath + "pwff_grn_.grn_x.npy")
x = torch.from_numpy(x).to(device)
x.requires_grad = True;

input_dim = 64
hidden_dim = 64
output_dim = 64
context_dim = None
dropout = 0.0
batch_first = True

grn = GatedResidualNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, context_dim=context_dim, batch_first=batch_first, use_mycaffe=True, debug=True, tag="test")
grn.to(device)

for param in grn.state_dict():
    DebugFunction.trace(grn.state_dict()[param], "test_grn_" + param)

DebugFunction.trace(x, "test_grn_x")
x = debug(x)

y = grn(x)

DebugFunction.trace(y, "test_grn_y")
y = debug(y)

y_grad = np.load(strPath + "pwff_grn_.grn_y.grad.npy")
y_grad = torch.from_numpy(y_grad).to(device)

y.backward(y_grad)

#DebugFunction.trace(grn.skip_layer.module.weight.grad, "test_grn.skip_layer.module.weight.grad")
#DebugFunction.trace(grn.skip_layer.module.bias.grad, "test_grn.skip_layer.module.bias.grad")
DebugFunction.trace(grn.fc1.module.weight.grad, "test_grn.fc1.module.weight.grad")
DebugFunction.trace(grn.fc1.module.bias.grad, "test_grn.fc1.module.bias.grad")
DebugFunction.trace(grn.fc2.module.weight.grad, "test_grn.fc2.module.weight.grad")
DebugFunction.trace(grn.fc2.module.bias.grad, "test_grn.fc2.module.bias.grad")
DebugFunction.trace(grn.gate.module.fc1.weight.grad, "test_grn.gate.module.fc1.weight.grad")
DebugFunction.trace(grn.gate.module.fc1.bias.grad, "test_grn.gate.module.fc1.bias.grad")
DebugFunction.trace(grn.gate.module.fc2.weight.grad, "test_grn.gate.module.fc2.weight.grad")
DebugFunction.trace(grn.gate.module.fc2.bias.grad, "test_grn.gate.module.fc2.bias.grad")

print("done!");


