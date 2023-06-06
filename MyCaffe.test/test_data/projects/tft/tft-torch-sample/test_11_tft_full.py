import copy
import os
cwd = os.getcwd()

import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork, GatedLinearUnit, GateAddNorm, InterpretableMultiHeadAttention, TemporalFusionTransformer
from tft_torch.base_blocks import TimeDistributed, NullTransform
from tft_helper import apply_temporal_selection, apply_self_attention
from omegaconf import OmegaConf,DictConfig
import torch.nn.init as init

seed = 1704
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

path = "full";
debug = DebugFunction.apply
DebugFunction.set_output_path(path, 0)

os.chdir(cwd)
print("\n" + os.getcwd())

strPathWt = "data/favorita/weights/"
strPath = "test/iter_0/"
strPathBase = "test/full/iter_0/data/"

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

num_hist_categorical = 7
hist_categorical_cardinalities = [2,3,8,13,72,6,28]
num_hist_numeric = 4

num_future_categorical = 7
future_categorical_cardinalities = [2,3,8,13,72,6,28]
num_future_numeric = 1

num_static_numeric = 0
static_categorical_cardinalities = [54, 3627, 23, 17, 6, 18, 33, 320, 3]
num_static_categorical = 9

num_historical_steps = 90
num_future_steps = 30
target_window_start = None
target_window_start_idx = (target_window_start - 1) if target_window_start is not None else 0

data_props = {}
data_props['num_historical_numeric'] = num_hist_numeric
data_props['num_historical_categorical'] = num_hist_categorical
data_props['historical_categorical_cardinalities'] = hist_categorical_cardinalities
data_props['num_future_numeric'] = num_future_numeric
data_props['num_future_categorical'] = num_future_categorical
data_props['future_categorical_cardinalities'] = future_categorical_cardinalities
data_props['num_static_numeric'] = num_static_numeric
data_props['num_static_categorical'] = num_static_categorical
data_props['static_categorical_cardinalities'] = static_categorical_cardinalities

configuration = {'optimization':
                 {
                     'batch_size': {'training': 16, 'inference': 16},
                     'learning_rate': 0.001,
                     'max_grad_norm': 0.0,
                 }
                 ,
                 'model':
                 {
                     'dropout': 0.0, # 0.05 _change,
                     'state_size': 64,
                     'output_quantiles': [0.1, 0.5, 0.9],
                     'lstm_layers': 2,
                     'attention_heads': 4
                 },
                 # these arguments are related to possible extensions of the model class
                 'task_type':'regression',
                 'target_window_start': None
                }
configuration['data_props'] = data_props

model = TemporalFusionTransformer(config=OmegaConf.create(configuration), debug=True, use_mycaffe=True, lstm_use_mycaffe=True, tag="tft.full")

# initialize the weights of the model
def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                bias.data[:n // 3].fill_(-1.)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

model.apply(weight_init)

param_list = []
for param in model.state_dict():
        DebugFunction.trace(model.state_dict()[param], "tft.full." + param, "weights")     
        param_list.append(param)
model.to(device)

batch = {}
static_feats_numeric = np.load(strPathBase + "0_static_feats_numeric.npy")
static_feats_numeric = torch.from_numpy(static_feats_numeric).to(device)
static_feats_categorical = np.load(strPathBase + "0_static_feats_categorical.npy")
static_feats_categorical = torch.from_numpy(static_feats_categorical).to(device)
historical_ts_numeric = np.load(strPathBase + "0_historical_ts_numeric.npy")
historical_ts_numeric = torch.from_numpy(historical_ts_numeric).to(device)
historical_ts_categorical = np.load(strPathBase + "0_historical_ts_categorical.npy")
historical_ts_categorical = torch.from_numpy(historical_ts_categorical).to(device)
future_ts_numeric = np.load(strPathBase + "0_future_ts_numeric.npy")
future_ts_numeric = torch.from_numpy(future_ts_numeric).to(device)
future_ts_categorical = np.load(strPathBase + "0_future_ts_categorical.npy")
future_ts_categorical = torch.from_numpy(future_ts_categorical).to(device)
target = np.load(strPathBase + "0_target.npy")
target = torch.from_numpy(target).to(device)

batch['static_feats_numeric'] = static_feats_numeric
batch['static_feats_categorical'] = static_feats_categorical
batch['historical_ts_numeric'] = historical_ts_numeric
batch['historical_ts_categorical'] = historical_ts_categorical
batch['future_ts_numeric'] = future_ts_numeric
batch['future_ts_categorical'] = future_ts_categorical
batch['target'] = target

outputs = model(batch)

model.past_lstm.save_wts("tft.full.", "weights")
model.future_lstm.save_wts("tft.full.", "weights")

predicted_quantiles = outputs['predicted_quantiles']

grad = np.load(strPathBase + "tft.all.predicted_quantiles.grad.npy")
grad = torch.from_numpy(grad).to(device)

predicted_quantiles.backward(grad)

print("done!");


