import os
cwd = os.getcwd()
import numpy as np
import torch
from torch import nn
from typing import Dict
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding

os.chdir(cwd)
print("\n" + os.getcwd())

strSubPath = "ice_fut"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPathWt = "data/favorita/weights/future_ts_transform/"
strPath = "data/favorita/batch256/batch_0"
batch = load_batch(strPath)

empty_tensor = torch.empty((0, 0))
x_categorical = batch.get('future_ts_categorical', empty_tensor)
empty_tensor = torch.empty((0, 0))
x_numeric = batch.get('future_ts_numeric', empty_tensor)

num_categorical = 7
categorical_cardinalities = [2,3,8,13,72,6,28]

num_numeric = 1
state_size = 64

future_ts_transform = InputChannelEmbedding(
            state_size=state_size,
            num_numeric=num_numeric,
            num_categorical=num_categorical,
            categorical_cardinalities=categorical_cardinalities,
            time_distribute=True, path=strSubPath, debug=True)

wt0 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.0.weight.npy")
b0 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.0.bias.npy")
future_ts_transform.numeric_transform.module.numeric_projection_layers[0].weight = nn.Parameter(torch.from_numpy(wt0))
future_ts_transform.numeric_transform.module.numeric_projection_layers[0].bias = nn.Parameter(torch.from_numpy(b0))
DebugFunction.trace(future_ts_transform.numeric_transform.module.numeric_projection_layers[0].weight, "future_proj_layer.0.weight")
DebugFunction.trace(future_ts_transform.numeric_transform.module.numeric_projection_layers[0].bias, "future_proj_layer.0.bias")

wt0 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.0.weight.npy")
future_ts_transform.categorical_transform.module.categorical_embedding_layers[0].weight = nn.Parameter(torch.from_numpy(wt0))
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[0].weight, "future_emb_layer.0.weight")

wt1 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.1.weight.npy")
future_ts_transform.categorical_transform.module.categorical_embedding_layers[1].weight = nn.Parameter(torch.from_numpy(wt1))
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[1].weight, "future_emb_layer.1.weight")

wt2 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.2.weight.npy")
future_ts_transform.categorical_transform.module.categorical_embedding_layers[2].weight = nn.Parameter(torch.from_numpy(wt2))
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[2].weight, "future_emb_layer.2.weight")

wt3 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.3.weight.npy")
future_ts_transform.categorical_transform.module.categorical_embedding_layers[3].weight = nn.Parameter(torch.from_numpy(wt3))
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[3].weight, "future_emb_layer.3.weight")

wt4 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.4.weight.npy")
future_ts_transform.categorical_transform.module.categorical_embedding_layers[4].weight = nn.Parameter(torch.from_numpy(wt4))
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[4].weight, "future_emb_layer.4.weight")

wt5 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.5.weight.npy")
future_ts_transform.categorical_transform.module.categorical_embedding_layers[5].weight = nn.Parameter(torch.from_numpy(wt5))
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[5].weight, "future_emb_layer.5.weight")

wt6 = np.load(strPathWt + "categorical_transform.module.categorical_embedding_layers.6.weight.npy")
future_ts_transform.categorical_transform.module.categorical_embedding_layers[6].weight = nn.Parameter(torch.from_numpy(wt6))
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[6].weight, "future_emb_layer.6.weight")

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
future_ts_transform.to(device)
x_categorical = x_categorical.to(device)
x_numeric = x_numeric.to(device)

DebugFunction.trace(x_categorical, "x_categorical")
DebugFunction.trace(x_numeric, "x_numeric")

future_processed_input = future_ts_transform(x_numeric=x_numeric, x_categorical=x_categorical)

DebugFunction.trace(future_processed_input, "future_processed_input")
future_processed_input = debug(future_processed_input)

p0 = future_processed_input.clone()

p0 = p0 * 0 + 1

loss = (future_processed_input[0] - p0).sum()

DebugFunction.trace(loss, "input_future_loss")
loss = debug(loss)

loss.backward()

DebugFunction.trace(future_ts_transform.numeric_transform.module.numeric_projection_layers[0].weight.grad, "future_proj_layer.0.weight_grad")
DebugFunction.trace(future_ts_transform.numeric_transform.module.numeric_projection_layers[0].bias.grad, "future_proj_layer.0.bias_grad")

DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[0].weight.grad, "future_emb_layer.0.weight_grad")
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[1].weight.grad, "future_emb_layer.1.weight_grad")
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[2].weight.grad, "future_emb_layer.2.weight_grad")
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[3].weight.grad, "future_emb_layer.3.weight_grad")
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[4].weight.grad, "future_emb_layer.4.weight_grad")
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[5].weight.grad, "future_emb_layer.5.weight_grad")
DebugFunction.trace(future_ts_transform.categorical_transform.module.categorical_embedding_layers[6].weight.grad, "future_emb_layer.6.weight_grad")

print("done!");


