import os
cwd = os.getcwd()

import torch
import torch.nn as nn
import numpy as np
from utility import DebugFunction
from torch_cfc import CfcCell, Cfc

DebugFunction.seed_everything()

in_features = 82
hidden_size = 256
out_features = 2
no_gate = False

os.chdir(cwd)
print(os.getcwd())

if no_gate:
    strSubPath = "cfc_no_gate"
else:
    strSubPath = "cfc_gate"

debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

x = np.load('test/x.npy')
x = torch.from_numpy(x).float()
x.requires_grad = True
timespans = np.load('test/timespans.npy')
timespans = torch.from_numpy(timespans).float()
timespans.requires_grad = True
mask = np.load('test/mask.npy')
mask = torch.from_numpy(mask).float()

BEST_DEFAULT = {
    "epochs": 57,
    "class_weight": 11.69,
    "clipnorm": 0,
    "hidden_size": 256,
    "base_lr": 0.002,
    "decay_lr": 0.9,
    "backbone_activation": "relu",
    "backbone_units": 64,
    "backbone_dr": 0.0  ,
    "backbone_layers": 2,
    "weight_decay": 4e-06,
    "optim": "adamw",
    "init": 0.5,
    "batch_size": 128,
    "use_mixed": False,
    "no_gate": no_gate,
    "minimal": False,
    "use_ltc": False,
    "debug": True
}

DebugFunction.trace(x, "x")
DebugFunction.trace(timespans, "timespans")
DebugFunction.trace(mask, "mask")

x = debug(x)
timespans = debug(timespans)
mask = debug(mask)

cfc = Cfc(in_features, hidden_size, out_features, BEST_DEFAULT, False, False, False)
cfc.save_weights(strSubPath, "cfc.")

y = cfc(x, timespans, mask)

DebugFunction.trace(y, "y")
y = debug(y)

y.sum().backward()

print("done!")

