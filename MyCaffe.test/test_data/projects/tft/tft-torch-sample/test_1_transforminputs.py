import os
cwd = os.getcwd()

import copy
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork, GateAddNorm
from tft_torch.base_blocks import TimeDistributed, NullTransform
from tft_helper import apply_static_enrichment

os.chdir(cwd)
print(os.getcwd())

strSubPath = "ti"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPathWt = "data/favorita/weights/"
strPath = "test/iter_0.base_set/";

state_size = 64
num_static_numeric = 0
num_static_categorical = 9
static_categorical_cardinalities = [54, 3627, 23, 17, 6, 18, 33, 320, 3]

num_historical_numeric = 4
num_historical_categorical = 7
historical_categorical_cardinalities = [2, 3, 8, 13, 72, 6, 28]

num_future_numeric = 1
num_future_categorical = 7
future_categorical_cardinalities = [2, 3, 8, 13, 72, 6, 28]

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

static_transform = InputChannelEmbedding(state_size=state_size,
                                                      num_numeric=num_static_numeric,
                                                      num_categorical=num_static_categorical,
                                                      categorical_cardinalities=static_categorical_cardinalities,
                                                      time_distribute=False, debug=True, tag = "tft.ti.static")

historical_ts_transform = InputChannelEmbedding(
            state_size=state_size,
            num_numeric=num_historical_numeric,
            num_categorical=num_historical_categorical,
            categorical_cardinalities=historical_categorical_cardinalities,
            time_distribute=True, debug=True, tag = "tft.ti.historical")

future_ts_transform = InputChannelEmbedding(
            state_size=state_size,
            num_numeric=num_future_numeric,
            num_categorical=num_future_categorical,
            categorical_cardinalities=future_categorical_cardinalities,
            time_distribute=True, debug=True, tag = "tft.ti.future")

for param in static_transform.state_dict():
        strFile = strPathWt + "static_transform/" + param + ".npy"
        data = np.load(strFile)
        static_transform.state_dict()[param] = torch.from_numpy(data).to(device)        
        DebugFunction.trace(static_transform.state_dict()[param], "tft.ti.static." + param)
static_transform.to(device)

for param in historical_ts_transform.state_dict():
        strFile = strPathWt + "hist_ts_transform/" + param + ".npy"
        data = np.load(strFile)
        historical_ts_transform.state_dict()[param] = torch.from_numpy(data).to(device)        
        DebugFunction.trace(historical_ts_transform.state_dict()[param], "tft.ti.historical." + param)
historical_ts_transform.to(device)

for param in future_ts_transform.state_dict():
        strFile = strPathWt + "future_ts_transform/" + param + ".npy"
        data = np.load(strFile)
        future_ts_transform.state_dict()[param] = torch.from_numpy(data).to(device)        
        DebugFunction.trace(future_ts_transform.state_dict()[param], "tft.ti.future." + param)
future_ts_transform.to(device)

x_numeric_static = np.load(strPath + "tft.static.x_numeric.npy")
x_numeric_static = torch.from_numpy(x_numeric_static).to(device)
x_categorical_static = np.load(strPath + "tft.static.x_categorical.npy")
x_categorical_static = torch.from_numpy(x_categorical_static).to(device)

x_numeric_hist = np.load(strPath + "tft.historical.x_numeric.npy")
x_numeric_hist = torch.from_numpy(x_numeric_hist).to(device)
x_numeric_hist.requires_grad = True
x_categorical_hist = np.load(strPath + "tft.historical.x_categorical.npy")
x_categorical_hist = torch.from_numpy(x_categorical_hist).to(device)

x_numeric_future = np.load(strPath + "tft.future.x_numeric.npy")
x_numeric_future = torch.from_numpy(x_numeric_future).to(device)
x_numeric_future.requires_grad = True
x_categorical_future = np.load(strPath + "tft.future.x_categorical.npy")
x_categorical_future = torch.from_numpy(x_categorical_future).to(device)

static_rep = static_transform(x_numeric_static, x_categorical_static)
historical_ts_rep = historical_ts_transform(x_numeric_hist, x_categorical_hist)
future_ts_rep = future_ts_transform(x_numeric_future, x_categorical_future)

DebugFunction.trace(static_rep, "tft.ti.static_rep")
DebugFunction.trace(historical_ts_rep, "tft.ti.historical_ts_rep")
DebugFunction.trace(future_ts_rep, "tft.ti.future_ts_rep")

static_rep_grad = np.load(strPath + "tft.static.merged_transformations.grad.npy")
static_rep_grad = torch.from_numpy(static_rep_grad).to(device)
static_rep.backward(static_rep_grad)

historical_rep_grad = np.load(strPath + "tft.historical.merged_transformations.grad.npy")
historical_rep_grad = torch.from_numpy(historical_rep_grad).to(device)
historical_ts_rep.backward(historical_rep_grad)

future_rep_grad = np.load(strPath + "tft.future.merged_transformations.grad.npy")
future_rep_grad = torch.from_numpy(future_rep_grad).to(device)
future_ts_rep.backward(future_rep_grad)

for i in range(num_static_categorical):
    DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[i].weight, "tft.ti.static.categorical_transform.categorical_embedding_layers.%d.weight.grad" % (i))

for i in range(num_historical_categorical):
    DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[i].weight, "tft.ti.historical.categorical_transform.module.categorical_embedding_layers.%d.weight.grad" % (i))

for i in range(num_historical_numeric):
    DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[i].weight, "tft.ti.historical.numeric_transform.module.numeric_projection_layers.%d.weight.grad" % (i))
    DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[i].bias, "tft.ti.historical.numeric_transform.module.numeric_projection_layers.%d.bias.grad" % (i))

for i in range(num_future_categorical):
    DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[i].weight, "tft.ti.future.categorical_transform.module.categorical_embedding_layers.%d.weight.grad" % (i))

for i in range(num_future_numeric):
    DebugFunction.trace(future_ts_transform.numeric_transform.module.numeric_projection_layers[i].weight, "tft.ti.future.numeric_transform.module.numeric_projection_layers.%d.weight.grad" % (i))
    DebugFunction.trace(future_ts_transform.numeric_transform.module.numeric_projection_layers[i].bias, "tft.ti.future.numeric_transform.module.numeric_projection_layers.%d.bias.grad" % (i))

print("done!");


