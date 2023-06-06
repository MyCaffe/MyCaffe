import os
cwd = os.getcwd()

import copy
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork
from tft_torch.base_blocks import TimeDistributed, NullTransform

os.chdir(cwd)
print(os.getcwd())

strSubPath = "vsn_stat"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPath = "test/iter_0.base_set/"
strPathWt = "data/favorita/weights/"

num_categorical = 9
categorical_cardinalities = [54, 3627, 23, 17, 6, 18, 33, 320, 3]
dropout = 0.0
num_numeric = 0
state_size = 64

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

static_selection = VariableSelectionNetwork(
            input_dim=state_size,
            num_inputs=num_numeric + num_categorical,
            hidden_dim=state_size, dropout=dropout, debug=True, use_mycaffe=True, tag="tft.vsn.static", path=strSubPath)

for param in static_selection.state_dict():
        strFile = strPathWt + "static_selection/" + param + ".npy"
        data = np.load(strFile)
        static_selection.state_dict()[param] = torch.from_numpy(data).to(device)    
        DebugFunction.trace(static_selection.state_dict()[param], "tft.vsn.static." + param)

static_selection.to(device)

static_rep = np.load(strPath + "tft.static_rep.npy")
static_rep = torch.from_numpy(static_rep).to(device)
static_rep.requires_grad = True

DebugFunction.trace(static_rep, "ZZZ.vsn.static_rep")
static_rep = debug(static_rep)

selected_static, selected_static_wts = static_selection(static_rep)

DebugFunction.trace(selected_static, "ZZZ.vsn.selected_static")
DebugFunction.trace(selected_static_wts, "ZZZ.vsn.selected_static_wts")
selected_static = debug(selected_static)
selected_static_wts = debug(selected_static_wts)

grad = np.load(strPath + "tft.selected_static.grad.npy")
grad = torch.from_numpy(grad).to(device)

DebugFunction.trace(static_rep, "ZZZ.vsn.selected_static.grad")

selected_static.backward(grad)

print("done!");


