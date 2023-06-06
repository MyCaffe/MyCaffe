import os
cwd = os.getcwd()

import os
import pickle
from typing import Dict,List,Tuple
from functools import partial
import copy
import numpy as np
from omegaconf import OmegaConf,DictConfig
import pandas as pd
from tqdm import tqdm
import torch
from torch import optim
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Subset
from tft_torch.tft import TemporalFusionTransformer
import tft_torch.loss as tft_loss
from utility import save_batch, save_weights
from pathlib import Path
from pandas import Timestamp
import pandas as pd
import datetime

os.chdir(cwd)
print(os.getcwd())

nMax = 10000 # set to max size for dataset clipping, set to 0 for full dataset size.

print(torch.__version__)
ndev = torch.cuda.device_count()
print(ndev)
torch.cuda.set_device(0)
ncur = torch.cuda.current_device()
print(ncur)
strName = torch.cuda.get_device_name(0)
print(strName)

def convertToUnixTimeValue(data):
    rgv = []
    for item in data:
        rgv.append(item.asm8.astype('datetime64[s]').astype(int))
    return rgv
        
# Set the path to the location where we saved the data generated using 'dataset_creation.py'
data_path = 'data/favorita/data.pickle'

# Read the pickle file and take a pick at its content.
with open(data_path,'rb') as fp:
    data = pickle.load(fp)

if nMax > 0:
    strType = 'train'
    data['data_sets'][strType]['time_index'] = data['data_sets'][strType]['time_index'][:nMax]
    data['data_sets'][strType]['combination_id'] = data['data_sets'][strType]['combination_id'][:nMax]
    data['data_sets'][strType]['static_feats_numeric'] = data['data_sets'][strType]['static_feats_numeric'][:nMax]
    data['data_sets'][strType]['static_feats_categorical'] = data['data_sets'][strType]['static_feats_categorical'][:nMax]
    data['data_sets'][strType]['historical_ts_numeric'] = data['data_sets'][strType]['historical_ts_numeric'][:nMax]
    data['data_sets'][strType]['historical_ts_categorical'] = data['data_sets'][strType]['historical_ts_categorical'][:nMax]
    data['data_sets'][strType]['future_ts_numeric'] = data['data_sets'][strType]['future_ts_numeric'][:nMax]
    data['data_sets'][strType]['future_ts_categorical'] = data['data_sets'][strType]['future_ts_categorical'][:nMax]
    data['data_sets'][strType]['target'] = data['data_sets'][strType]['target'][:nMax]
    strType = 'validation'
    data['data_sets'][strType]['time_index'] = data['data_sets'][strType]['time_index'][:nMax]
    data['data_sets'][strType]['combination_id'] = data['data_sets'][strType]['combination_id'][:nMax]
    data['data_sets'][strType]['static_feats_numeric'] = data['data_sets'][strType]['static_feats_numeric'][:nMax]
    data['data_sets'][strType]['static_feats_categorical'] = data['data_sets'][strType]['static_feats_categorical'][:nMax]
    data['data_sets'][strType]['historical_ts_numeric'] = data['data_sets'][strType]['historical_ts_numeric'][:nMax]
    data['data_sets'][strType]['historical_ts_categorical'] = data['data_sets'][strType]['historical_ts_categorical'][:nMax]
    data['data_sets'][strType]['future_ts_numeric'] = data['data_sets'][strType]['future_ts_numeric'][:nMax]
    data['data_sets'][strType]['future_ts_categorical'] = data['data_sets'][strType]['future_ts_categorical'][:nMax]
    data['data_sets'][strType]['target'] = data['data_sets'][strType]['target'][:nMax]
    strType = 'test'
    data['data_sets'][strType]['time_index'] = data['data_sets'][strType]['time_index'][:nMax]
    data['data_sets'][strType]['combination_id'] = data['data_sets'][strType]['combination_id'][:nMax]
    data['data_sets'][strType]['static_feats_numeric'] = data['data_sets'][strType]['static_feats_numeric'][:nMax]
    data['data_sets'][strType]['static_feats_categorical'] = data['data_sets'][strType]['static_feats_categorical'][:nMax]
    data['data_sets'][strType]['historical_ts_numeric'] = data['data_sets'][strType]['historical_ts_numeric'][:nMax]
    data['data_sets'][strType]['historical_ts_categorical'] = data['data_sets'][strType]['historical_ts_categorical'][:nMax]
    data['data_sets'][strType]['future_ts_numeric'] = data['data_sets'][strType]['future_ts_numeric'][:nMax]
    data['data_sets'][strType]['future_ts_categorical'] = data['data_sets'][strType]['future_ts_categorical'][:nMax]
    data['data_sets'][strType]['target'] = data['data_sets'][strType]['target'][:nMax]

data_path = os.getcwd() + "\\data\\favorita\\raw"
# set parent directory as the output path
output_path = Path(data_path).parent.absolute()

if (nMax > 0):
    with open(os.path.join(output_path, 'data.small.%d.pickle' % (nMax)), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

strType = 'train'
strFile = os.path.join(output_path, strType + "_" + "time_index.npy")
np.save(strFile, convertToUnixTimeValue(data['data_sets'][strType]['time_index']))
strFile = os.path.join(output_path, strType + "_" + "combination_id.npy")
np.save(strFile, data['data_sets'][strType]['combination_id'])
strFile = os.path.join(output_path, strType + "_" + "static_feats_numeric.npy")
np.save(strFile, data['data_sets'][strType]['static_feats_numeric'])
strFile = os.path.join(output_path, strType + "_" + "static_feats_categorical.npy")
np.save(strFile, data['data_sets'][strType]['static_feats_categorical'])
strFile = os.path.join(output_path, strType + "_" + "historical_ts_numeric.npy")
np.save(strFile, data['data_sets'][strType]['historical_ts_numeric'])
strFile = os.path.join(output_path, strType + "_" + "historical_ts_categorical.npy")
np.save(strFile, data['data_sets'][strType]['historical_ts_categorical'])
strFile = os.path.join(output_path, strType + "_" + "future_ts_numeric.npy")
np.save(strFile, data['data_sets'][strType]['future_ts_numeric'])
strFile = os.path.join(output_path, strType + "_" + "future_ts_categorical.npy")
np.save(strFile, data['data_sets'][strType]['future_ts_categorical'])
strFile = os.path.join(output_path, strType + "_" + "target.npy")
np.save(strFile, data['data_sets'][strType]['target'])
strType = 'validation'
strFile = os.path.join(output_path, strType + "_" + "time_index.npy")
np.save(strFile, convertToUnixTimeValue(data['data_sets'][strType]['time_index']))
strFile = os.path.join(output_path, strType + "_" + "combination_id.npy")
np.save(strFile, data['data_sets'][strType]['combination_id'])
strFile = os.path.join(output_path, strType + "_" + "static_feats_numeric.npy")
np.save(strFile, data['data_sets'][strType]['static_feats_numeric'])
strFile = os.path.join(output_path, strType + "_" + "static_feats_categorical.npy")
np.save(strFile, data['data_sets'][strType]['static_feats_categorical'])
strFile = os.path.join(output_path, strType + "_" + "historical_ts_numeric.npy")
np.save(strFile, data['data_sets'][strType]['historical_ts_numeric'])
strFile = os.path.join(output_path, strType + "_" + "historical_ts_categorical.npy")
np.save(strFile, data['data_sets'][strType]['historical_ts_categorical'])
strFile = os.path.join(output_path, strType + "_" + "future_ts_numeric.npy")
np.save(strFile, data['data_sets'][strType]['future_ts_numeric'])
strFile = os.path.join(output_path, strType + "_" + "future_ts_categorical.npy")
np.save(strFile, data['data_sets'][strType]['future_ts_categorical'])
strFile = os.path.join(output_path, strType + "_" + "target.npy")
np.save(strFile, data['data_sets'][strType]['target'])
strType = 'test'
strFile = os.path.join(output_path, strType + "_" + "time_index.npy")
np.save(strFile, convertToUnixTimeValue(data['data_sets'][strType]['time_index']))
strFile = os.path.join(output_path, strType + "_" + "combination_id.npy")
np.save(strFile, data['data_sets'][strType]['combination_id'])
strFile = os.path.join(output_path, strType + "_" + "static_feats_numeric.npy")
np.save(strFile, data['data_sets'][strType]['static_feats_numeric'])
strFile = os.path.join(output_path, strType + "_" + "static_feats_categorical.npy")
np.save(strFile, data['data_sets'][strType]['static_feats_categorical'])
strFile = os.path.join(output_path, strType + "_" + "historical_ts_numeric.npy")
np.save(strFile, data['data_sets'][strType]['historical_ts_numeric'])
strFile = os.path.join(output_path, strType + "_" + "historical_ts_categorical.npy")
np.save(strFile, data['data_sets'][strType]['historical_ts_categorical'])
strFile = os.path.join(output_path, strType + "_" + "future_ts_numeric.npy")
np.save(strFile, data['data_sets'][strType]['future_ts_numeric'])
strFile = os.path.join(output_path, strType + "_" + "future_ts_categorical.npy")
np.save(strFile, data['data_sets'][strType]['future_ts_categorical'])
strFile = os.path.join(output_path, strType + "_" + "target.npy")
np.save(strFile, data['data_sets'][strType]['target'])

print("done!")
