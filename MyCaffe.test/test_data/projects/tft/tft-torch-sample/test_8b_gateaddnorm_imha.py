import os
cwd = os.getcwd()
import copy
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork, GatedLinearUnit, GateAddNorm
from tft_torch.base_blocks import TimeDistributed, NullTransform
from tft_helper import apply_temporal_selection

os.chdir(cwd)
print(os.getcwd())

strSubPath = "imha"
strPathWt = "data/favorita/weights/"
strPath = "test/imha/iter_0/"

debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

x = np.load(strPath + "tft.asa.gan_.gan_x.npy")
x = torch.from_numpy(x).to(device)
x.requires_grad = True;

input_dim = 64
dropout = 0.0

gan = GateAddNorm(input_dim=input_dim, dropout=dropout, use_mycaffe=True, debug=True, tag="test")
gan.to(device)

gan.gate.module.fc1.weight = torch.nn.Parameter(torch.from_numpy(np.load(strPath + "tft.asa.gan_gan_glu.internal.fc1.weight.npy")).to(device))
gan.gate.module.fc1.bias = torch.nn.Parameter(torch.from_numpy(np.load(strPath + "tft.asa.gan_gan_glu.internal.fc1.bias.npy")).to(device))
gan.gate.module.fc2.weight = torch.nn.Parameter(torch.from_numpy(np.load(strPath + "tft.asa.gan_gan_glu.internal.fc2.weight.npy")).to(device))
gan.gate.module.fc2.bias = torch.nn.Parameter(torch.from_numpy(np.load(strPath + "tft.asa.gan_gan_glu.internal.fc2.bias.npy")).to(device))

y = gan(x)

y_grad = np.load(strPath + "tft.asa.gan_.gan_y.grad.npy");
y_grad = torch.from_numpy(y_grad).to(device)

y.backward(y_grad)

DebugFunction.trace(gan.gate.module.fc1.weight.grad, "tft.asa.gan_gate_gan_glu.internal.fc1.weight.grad")
DebugFunction.trace(gan.gate.module.fc1.bias.grad, "tft.asa.gan_gate_gan_glu.internal.fc1.bias.grad")
DebugFunction.trace(gan.gate.module.fc2.weight.grad, "tft.asa.gan_gate_gan_glu.internal.fc2.weight.grad")
DebugFunction.trace(gan.gate.module.fc2.bias.grad, "tft.asa.gan_gate_gan_glu.internal.fc2.bias.grad")

print("done!");


