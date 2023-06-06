import os
cwd = os.getcwd()
import numpy as np
import torch
from torch import nn
from typing import Dict
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding
from tft_torch.base_blocks import TimeDistributed, NullTransform

os.chdir(cwd)
print("\n" + os.getcwd())

strSubPath = "ice_hist"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPathWt = "data/favorita/weights/hist_ts_transform/"
strPath = "data/favorita/batch256/batch_0"
batch = load_batch(strPath)

empty_tensor = torch.empty((0, 0))
x_categorical = batch.get('historical_ts_categorical', empty_tensor)
empty_tensor = torch.empty((0, 0))
x_numeric = batch.get('historical_ts_numeric', empty_tensor)

num_categorical = 7
categorical_cardinalities = [2,3,8,13,72,6,28]

num_numeric = 4
state_size = 64

historical_ts_transform = InputChannelEmbedding(
            state_size=state_size,
            num_numeric=num_numeric,
            num_categorical=num_categorical,
            categorical_cardinalities=categorical_cardinalities,
            time_distribute=True, path=strSubPath, debug=True)

wt0 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.0.weight.npy")
b0 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.0.bias.npy")
historical_ts_transform.numeric_transform.module.numeric_projection_layers[0].weight = nn.Parameter(torch.from_numpy(wt0))
historical_ts_transform.numeric_transform.module.numeric_projection_layers[0].bias = nn.Parameter(torch.from_numpy(b0))
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[0].weight, "hist_proj_layer.0.weight")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[0].bias, "hist_proj_layer.0.bias")

wt1 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.1.weight.npy")
b1 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.1.bias.npy")
historical_ts_transform.numeric_transform.module.numeric_projection_layers[1].weight = nn.Parameter(torch.from_numpy(wt1))
historical_ts_transform.numeric_transform.module.numeric_projection_layers[1].bias = nn.Parameter(torch.from_numpy(b1))
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[1].weight, "hist_proj_layer.1.weight")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[1].bias, "hist_proj_layer.1.bias")

wt2 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.2.weight.npy")
b2 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.2.bias.npy")
historical_ts_transform.numeric_transform.module.numeric_projection_layers[2].weight = nn.Parameter(torch.from_numpy(wt2))
historical_ts_transform.numeric_transform.module.numeric_projection_layers[2].bias = nn.Parameter(torch.from_numpy(b2))
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[2].weight, "hist_proj_layer.2.weight")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[2].bias, "hist_proj_layer.2.bias")

wt3 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.3.weight.npy")
b3 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.3.bias.npy")
historical_ts_transform.numeric_transform.module.numeric_projection_layers[3].weight = nn.Parameter(torch.from_numpy(wt3))
historical_ts_transform.numeric_transform.module.numeric_projection_layers[3].bias = nn.Parameter(torch.from_numpy(b3))
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[3].weight, "hist_proj_layer.3.weight")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[3].bias, "hist_proj_layer.3.bias")

wt0 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.0.weight.npy")
historical_ts_transform.categorical_transform.module.categorical_embedding_layers[0].weight = nn.Parameter(torch.from_numpy(wt0))
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[0].weight, "hist_emb_layer.0.weight")

wt1 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.1.weight.npy")
historical_ts_transform.categorical_transform.module.categorical_embedding_layers[1].weight = nn.Parameter(torch.from_numpy(wt1))
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[1].weight, "hist_emb_layer.1.weight")

wt2 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.2.weight.npy")
historical_ts_transform.categorical_transform.module.categorical_embedding_layers[2].weight = nn.Parameter(torch.from_numpy(wt2))
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[2].weight, "hist_emb_layer.2.weight")

wt3 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.3.weight.npy")
historical_ts_transform.categorical_transform.module.categorical_embedding_layers[3].weight = nn.Parameter(torch.from_numpy(wt3))
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[3].weight, "hist_emb_layer.3.weight")

wt4 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.4.weight.npy")
historical_ts_transform.categorical_transform.module.categorical_embedding_layers[4].weight = nn.Parameter(torch.from_numpy(wt4))
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[4].weight, "hist_emb_layer.4.weight")

wt5 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.5.weight.npy")
historical_ts_transform.categorical_transform.module.categorical_embedding_layers[5].weight = nn.Parameter(torch.from_numpy(wt5))
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[5].weight, "hist_emb_layer.5.weight")

wt6 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.6.weight.npy")
historical_ts_transform.categorical_transform.module.categorical_embedding_layers[6].weight = nn.Parameter(torch.from_numpy(wt6))
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[6].weight, "hist_emb_layer.6.weight")

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
historical_ts_transform.to(device)
x_categorical = x_categorical.to(device)
x_numeric = x_numeric.to(device)

DebugFunction.trace(x_categorical, "x_categorical")
DebugFunction.trace(x_numeric, "x_numeric")

hist_processed_input = historical_ts_transform(x_numeric=x_numeric, x_categorical=x_categorical)

DebugFunction.trace(hist_processed_input, "hist_processed_input")
hist_processed_input = debug(hist_processed_input)

p0 = hist_processed_input.clone()

p0 = p0 * 0 + 1

loss = (hist_processed_input[0] - p0).sum()

DebugFunction.trace(loss, "input_hist_loss")
loss = debug(loss)

loss.backward()

DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[0].weight.grad, "hist_proj_layer.0.weight.grad")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[0].bias.grad, "hist_proj_layer.0.bias.grad")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[1].weight.grad, "hist_proj_layer.1.weight.grad")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[1].bias.grad, "hist_proj_layer.1.bias.grad")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[2].weight.grad, "hist_proj_layer.2.weight.grad")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[2].bias.grad, "hist_proj_layer.2.bias.grad")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[3].weight.grad, "hist_proj_layer.3.weight.grad")
DebugFunction.trace(historical_ts_transform.numeric_transform.module.numeric_projection_layers[3].bias.grad, "hist_proj_layer.3.bias.grad")

DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[0].weight.grad, "hist_emb_layer.0.weight.grad")
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[1].weight.grad, "hist_emb_layer.1.weight.grad")
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[2].weight.grad, "hist_emb_layer.2.weight.grad")
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[3].weight.grad, "hist_emb_layer.3.weight.grad")
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[4].weight.grad, "hist_emb_layer.4.weight.grad")
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[5].weight.grad, "hist_emb_layer.5.weight.grad")
DebugFunction.trace(historical_ts_transform.categorical_transform.module.categorical_embedding_layers[6].weight.grad, "hist_emb_layer.6.weight.grad")

print("done!");


