import os
cwd = os.getcwd()

import copy
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork, GateAddNorm, LstmEx
from tft_torch.base_blocks import TimeDistributed, NullTransform
from tft_helper import apply_sequential_processing, get_cudnn_lstm_weights

os.chdir(cwd)
print(os.getcwd())

strSubPath = "seqproc"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPathWt = "data/favorita/weights/"

num_categorical = 9
categorical_cardinalities = [2,3,8,13,72,6,28]
dropout = 0.0
num_numeric = 0
state_size = 64
lstm_layers = 2

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

static_covariate_encoder = GatedResidualNetwork(input_dim=state_size,
                                                        hidden_dim=state_size,
                                                        output_dim=state_size,
                                                        dropout=dropout, use_mycaffe=True, debug=True, path=strSubPath)
static_encoder_sequential_cell_init = copy.deepcopy(static_covariate_encoder)
static_encoder_sequential_state_init = copy.deepcopy(static_covariate_encoder)

past_lstm = LstmEx(tag="YYY.past_lstm", 
                   state=state_size,
                   num=lstm_layers,
                   debug=True,
                   use_mycaffe=True)

future_lstm = LstmEx(tag="YYY.future_lstm", 
                   state=state_size,
                   num=lstm_layers,
                   debug=True,
                   use_mycaffe=True)

post_lstm_gating = GateAddNorm(input_dim=state_size, dropout=dropout, debug = True, tag="test4")

idx = 0
for param in static_encoder_sequential_cell_init.state_dict():
        strFile = strPathWt + "static_encoder_sequential_cell_init/" + param + ".npy"
        data = np.load(strFile)
        static_encoder_sequential_cell_init.state_dict()[param] = torch.from_numpy(data).to(device)     
        DebugFunction.trace(static_encoder_sequential_cell_init.state_dict()[param], "tft.static_encoder_sequential_cell_init." + param)
        idx = idx + 1
static_encoder_sequential_cell_init.to(device)

idx = 0
for param in static_encoder_sequential_state_init.state_dict():
        strFile = strPathWt + "static_encoder_sequential_state_init/" + param + ".npy"
        data = np.load(strFile)
        static_encoder_sequential_state_init.state_dict()[param] = torch.from_numpy(data).to(device)        
        DebugFunction.trace(static_encoder_sequential_state_init.state_dict()[param], "tft.static_encoder_sequential_state_init." + param)
        idx = idx + 1
static_encoder_sequential_state_init.to(device)

idx = 0
for param in past_lstm.state_dict():
        strFile = strPathWt + "past_lstm/" + param + ".npy"
        #data = np.load(strFile)
        #past_lstm.state_dict()[param] = torch.from_numpy(data).to(device)        
        DebugFunction.trace(past_lstm.state_dict()[param], "ZZZ.tft.past_lstm." + param)
        idx = idx + 1
past_lstm.to(device)

idx = 0
for param in future_lstm.state_dict():
        strFile = strPathWt + "future_lstm/" + param + ".npy"
        #data = np.load(strFile)
        #future_lstm.state_dict()[param] = torch.from_numpy(data).to(device)        
        DebugFunction.trace(future_lstm.state_dict()[param], "ZZZ.tft.future_lstm." + param)
        idx = idx + 1
future_lstm.to(device)

idx = 0
for param in post_lstm_gating.state_dict():
        strFile = strPathWt + "post_lstm_gating/" + param + ".npy"
        data = np.load(strFile)
        post_lstm_gating.state_dict()[param] = torch.from_numpy(data).to(device)        
        DebugFunction.trace(post_lstm_gating.state_dict()[param], "tft.post_lstm_gating." + param)
        idx = idx + 1
post_lstm_gating.to(device)

selected_historical = np.load("test/iter_0.base_set/tft.all.asp.selected_historical.npy")
selected_historical = torch.from_numpy(selected_historical).to(device)
selected_historical.requires_grad = True

selected_future = np.load("test/iter_0.base_set/tft.all.asp.selected_future.npy")
selected_future = torch.from_numpy(selected_future).to(device)
selected_future.requires_grad = True

c_seq_cell = np.load("test/iter_0.base_set/tft.all.asp.c_seq_cell.npy")
c_seq_cell = torch.from_numpy(c_seq_cell).to(device)
c_seq_cell.requires_grad = True

c_seq_hidden = np.load("test/iter_0.base_set/tft.all.asp.c_seq_hidden.npy")
c_seq_hidden = torch.from_numpy(c_seq_hidden).to(device)
c_seq_hidden.requires_grad = True

gated_lstm_output = apply_sequential_processing(selected_historical=selected_historical,
                                                             selected_future=selected_future,
                                                             c_seq_hidden=c_seq_hidden,
                                                             c_seq_cell=c_seq_cell,
                                                             past_lstm=past_lstm, future_lstm=future_lstm, post_lstm_gating=post_lstm_gating, lstm_layers=lstm_layers, path=strSubPath)

DebugFunction.trace(gated_lstm_output, "gated_lstm_output")
gated_lstm_output = debug(gated_lstm_output)

gated_lstm_output_grad = np.load("test/iter_0.base_set/tft.all.asp.gated_lstm_output.grad.npy")
gated_lstm_output_grad = torch.from_numpy(gated_lstm_output_grad).to(device)

gated_lstm_output.backward(gated_lstm_output_grad)

print("done!");


