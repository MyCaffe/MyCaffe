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

strSubPath = "ice_stat"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPathWt = "data/favorita/weights/static_transform/"
strPath = "data/favorita/batch256/batch_0"
batch = load_batch(strPath)

empty_tensor = torch.empty((0, 0))
x_categorical = batch.get('static_feats_categorical', empty_tensor)
empty_tensor = torch.empty((0, 0))
x_numeric = batch.get('static_feats_numeric', empty_tensor)

num_categorical = 9
categorical_cardinalities = [54, 3627, 23, 17, 6, 18, 33, 320, 3]

num_numeric = 0
state_size = 64

static_transform = InputChannelEmbedding(
            state_size=state_size,
            num_numeric=num_numeric,
            num_categorical=num_categorical,
            categorical_cardinalities=categorical_cardinalities,
            time_distribute=False, path=strSubPath, debug=True)

wt0 = np.load(strPathWt + "categorical_transform.categorical_embedding_layers.0.weight.npy")
static_transform.categorical_transform.categorical_embedding_layers[0].weight = nn.Parameter(torch.from_numpy(wt0))
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[0].weight, "static_emb_layer.0.weight")

wt1 = np.load(strPathWt + "categorical_transform.categorical_embedding_layers.1.weight.npy")
static_transform.categorical_transform.categorical_embedding_layers[1].weight = nn.Parameter(torch.from_numpy(wt1))
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[1].weight, "static_emb_layer.1.weight")

wt2 = np.load(strPathWt + "categorical_transform.categorical_embedding_layers.2.weight.npy")
static_transform.categorical_transform.categorical_embedding_layers[2].weight = nn.Parameter(torch.from_numpy(wt2))
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[2].weight, "static_emb_layer.2.weight")

wt3 = np.load(strPathWt + "categorical_transform.categorical_embedding_layers.3.weight.npy")
static_transform.categorical_transform.categorical_embedding_layers[3].weight = nn.Parameter(torch.from_numpy(wt3))
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[3].weight, "static_emb_layer.3.weight")

wt4 = np.load(strPathWt + "categorical_transform.categorical_embedding_layers.4.weight.npy")
static_transform.categorical_transform.categorical_embedding_layers[4].weight = nn.Parameter(torch.from_numpy(wt4))
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[4].weight, "static_emb_layer.4.weight")

wt5 = np.load(strPathWt + "categorical_transform.categorical_embedding_layers.5.weight.npy")
static_transform.categorical_transform.categorical_embedding_layers[5].weight = nn.Parameter(torch.from_numpy(wt5))
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[5].weight, "static_emb_layer.5.weight")

wt6 = np.load(strPathWt + "categorical_transform.categorical_embedding_layers.6.weight.npy")
static_transform.categorical_transform.categorical_embedding_layers[6].weight = nn.Parameter(torch.from_numpy(wt6))
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[6].weight, "static_emb_layer.6.weight")

wt7 = np.load(strPathWt + "categorical_transform.categorical_embedding_layers.7.weight.npy")
static_transform.categorical_transform.categorical_embedding_layers[7].weight = nn.Parameter(torch.from_numpy(wt7))
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[7].weight, "static_emb_layer.7.weight")

wt8 = np.load(strPathWt + "categorical_transform.categorical_embedding_layers.8.weight.npy")
static_transform.categorical_transform.categorical_embedding_layers[8].weight = nn.Parameter(torch.from_numpy(wt8))
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[8].weight, "static_emb_layer.8.weight")

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
static_transform.to(device)
x_categorical = x_categorical.to(device)
x_numeric = x_numeric.to(device)

DebugFunction.trace(x_categorical, "x_categorical")
DebugFunction.trace(x_numeric, "x_numeric")

stat_processed_input = static_transform(x_numeric=x_numeric, x_categorical=x_categorical)

DebugFunction.trace(stat_processed_input, "stat_processed_input")
stat_processed_input = debug(stat_processed_input)

p0 = stat_processed_input.clone()

p0 = p0 * 0 + 1

loss = (stat_processed_input[0] - p0).sum()

DebugFunction.trace(loss, "input_stat_loss")
loss = debug(loss)

loss.backward()

DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[0].weight.grad, "static_emb_layer.0.weight_grad")
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[1].weight.grad, "static_emb_layer.1.weight_grad")
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[2].weight.grad, "static_emb_layer.2.weight_grad")
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[3].weight.grad, "static_emb_layer.3.weight_grad")
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[4].weight.grad, "static_emb_layer.4.weight_grad")
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[5].weight.grad, "static_emb_layer.5.weight_grad")
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[6].weight.grad, "static_emb_layer.6.weight_grad")
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[7].weight.grad, "static_emb_layer.7.weight_grad")
DebugFunction.trace(static_transform.categorical_transform.categorical_embedding_layers[8].weight.grad, "static_emb_layer.8.weight_grad")

print("done!");


