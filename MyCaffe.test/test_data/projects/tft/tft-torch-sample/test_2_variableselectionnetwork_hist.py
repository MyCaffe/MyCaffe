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

strSubPath = "vsn_hist"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPath = "test/iter_0.base_set/"
strPathWt = "data/favorita/weights/"

num_categorical = 7
categorical_cardinalities = [2,3,8,13,72,6,28]
num_numeric = 4
dropout = 0.0
state_size = 64

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

hist_selection = VariableSelectionNetwork(
            input_dim=state_size,
            num_inputs=num_numeric + num_categorical,
            hidden_dim=state_size, dropout=dropout, context_dim=state_size, debug=True, use_mycaffe=True, tag="tft.vsn.hist", path=strSubPath)

for param in hist_selection.state_dict():
        strFile = strPathWt + "hist_ts_selection/" + param + ".npy"
        data = np.load(strFile)
        hist_selection.state_dict()[param] = torch.from_numpy(data).to(device)    
        DebugFunction.trace(hist_selection.state_dict()[param], "tft.vsn.hist." + param)

hist_selection.to(device)

hist_rep = np.load(strPath + "tft.historical_ts_rep.npy")
hist_rep = torch.from_numpy(hist_rep).to(device)
hist_rep.requires_grad = True

c_selection = np.load(strPath + "tft.c_selection.npy")
c_selection = torch.from_numpy(c_selection).to(device)
c_selection.requires_grad = True

DebugFunction.trace(hist_rep, "tft.vsn.historical_rep1")
hist_rep = debug(hist_rep)

DebugFunction.trace(c_selection, "tft.vsn.c_selection1")
c_selection = debug(c_selection)

selected_hist, selected_hist_wts = apply_temporal_selection(hist_rep, c_selection, hist_selection, path=strSubPath, tag="ZZZ.vsn.historical")

DebugFunction.trace(selected_hist, "tft.vsn.selected_historical")
DebugFunction.trace(selected_hist_wts, "tft.vsn.selected_historical_wts")
selected_hist = debug(selected_hist)
selected_hist_wts = debug(selected_hist_wts)

grad = np.load(strPath + "tft.selected_historical.grad.npy")
grad = torch.from_numpy(grad).to(device)

selected_hist.backward(grad)

print("done!");


