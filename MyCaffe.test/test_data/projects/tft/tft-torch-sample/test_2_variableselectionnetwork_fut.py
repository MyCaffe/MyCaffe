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
from tft_helper import apply_temporal_selection

os.chdir(cwd)
print(os.getcwd())

strSubPath = "vsn_fut"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPath = "test/iter_0.base_set/"
strPathWt = "data/favorita/weights/"

num_categorical = 7
categorical_cardinalities = [2,3,8,13,72,6,28]
num_numeric = 1
dropout = 0.0
state_size = 64

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

future_selection = VariableSelectionNetwork(
            input_dim=state_size,
            num_inputs=num_numeric + num_categorical,
            hidden_dim=state_size, dropout=dropout, context_dim=state_size, debug=True, use_mycaffe=True, tag="tft.vsn.future", path=strSubPath)

for param in future_selection.state_dict():
        strFile = strPathWt + "future_ts_selection/" + param + ".npy"
        data = np.load(strFile)
        future_selection.state_dict()[param] = torch.from_numpy(data).to(device)    
        DebugFunction.trace(future_selection.state_dict()[param], "tft.vsn.future." + param)

future_selection.to(device)

future_rep = np.load(strPath + "tft.future_ts_rep.npy")
future_rep = torch.from_numpy(future_rep).to(device)
future_rep.requires_grad = True

c_selection = np.load(strPath + "tft.c_selection.npy")
c_selection = torch.from_numpy(c_selection).to(device)
c_selection.requires_grad = True

DebugFunction.trace(future_rep, "tft.vsn.future_rep1")
future_rep = debug(future_rep)

DebugFunction.trace(c_selection, "tft.vsn.c_selection1")
c_selection = debug(c_selection)

selected_future, selected_future_wts = apply_temporal_selection(future_rep, c_selection, future_selection, path=strSubPath, tag="ZZZ.vsn.future")

DebugFunction.trace(selected_future, "tft.vsn.selected_future")
DebugFunction.trace(selected_future_wts, "tft.vsn.selected_future_wts")
selected_future = debug(selected_future)
selected_future_wts = debug(selected_future_wts)

grad = np.load(strPath + "tft.selected_future.grad.npy")
grad = torch.from_numpy(grad).to(device)

selected_future.backward(grad)

print("done!");


