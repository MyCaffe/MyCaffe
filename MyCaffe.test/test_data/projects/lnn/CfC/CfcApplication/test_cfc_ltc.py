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
    strSubPath = "ltc_no_gate"
else:
    strSubPath = "ltc_gate"

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


BEST_LTC = {
    "optimizer": "adam",
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "forget_bias": 2.4,
    "epochs": 80,
    "class_weight": 8,
    "clipnorm": 0,
    "hidden_size": 64,
    "backbone_units": 64,
    "backbone_dr": 0.1,
    "backbone_layers": 3,
    "weight_decay": 0,
    "optim": "adamw",
    "init": 0.53,
    "batch_size": 64,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
    "debug" : True,
}

DebugFunction.trace(x, "x")
DebugFunction.trace(timespans, "timespans")
DebugFunction.trace(mask, "mask")

x = debug(x)
timespans = debug(timespans)
mask = debug(mask)

cfc = Cfc(in_features, hidden_size, out_features, BEST_LTC, False, False, False)
cfc.save_weights(strSubPath, "cfc.")

y = cfc(x, timespans, mask)

DebugFunction.trace(y, "y")
y = debug(y)

y.sum().backward()

print("done!")

