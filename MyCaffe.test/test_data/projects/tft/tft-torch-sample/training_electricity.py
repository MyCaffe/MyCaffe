import os
from pickletools import optimize
from tkinter import N
cwd = os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES']='1'

import pickle
from typing import Dict,List,Tuple
from functools import partial
import copy
import numpy as np
from omegaconf import OmegaConf,DictConfig
import pandas as pd
from tqdm import tqdm
import torch
import time
from datetime import datetime
from torch import optim
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Subset
from tft_torch.tft import TemporalFusionTransformer, LinearEx, LstmEx
from tft_torch.adamw import AdamW2
import tft_torch.loss as tft_loss
from utility import save_batch, save_weights, DebugFunction
from tft_torch.mycaffe import MyCaffe

seed = 1704
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

path = "all"

# Settings for debugging and generating data
#debug = True
#use_mycaffe = True
#use_mycaffe_data = False
#use_mycaffe_model_direct = False
#lstm_use_mycaffe = True
#use_mycaffe_model = False
#tag = "tft.all"
#test = False

# Settings for running
debug = False
use_mycaffe = False
use_mycaffe_data = False
use_mycaffe_model_direct = False
use_mycaffe_model = False
lstm_use_mycaffe = False
tag = "tft.all"
test = True

if use_mycaffe_data:
    mycaffe = MyCaffe(True)

os.chdir(cwd)

DebugFunction.set_output_path(path, 0)

print(torch.__version__)
ndev = torch.cuda.device_count()
print(ndev)
torch.cuda.set_device(0)
ncur = torch.cuda.current_device()
print(ncur)
strName = torch.cuda.get_device_name(0)
print(strName)


# Set the path to the location where we saved the data generated using 'dataset_creation.py'
data_path = 'data/favorita/data.pickle'

# Read the pickle file and take a pick at its content.
with open(data_path,'rb') as fp:
    data = pickle.load(fp)

list(data.keys())

# Display the content of the data_sets key, note that the shapes of the array, depend on the range of dates configured.
for set_name in data['data_sets']:
    print('=======')
    print(set_name)
    print('=======')
    for arr_name,arr in data['data_sets'][set_name].items():
        print(f"{arr_name} (shape,dtype)")
        print(arr.shape, arr.dtype)

# We have some configuration settings to make related to the optimization methodology and model structure.
configuration = {'optimization':
                 {
                     'batch_size': {'training': 64, 'inference': 256},
                     'learning_rate': 0.001,
                     'max_grad_norm': 0, #1.0,
                 }
                 ,
                 'model':
                 {
                     'dropout': 0.1,
                     'state_size': 64,
                     'output_quantiles': [0.1, 0.5, 0.9],
                     'lstm_layers': 2,
                     'attention_heads': 4
                 },
                 # these arguments are related to possible extensions of the model class
                 'task_type':'regression',
                 'target_window_start': None
                }

structure = {
    'num_historical_numeric': 4,            # power_usage, hour, day_of_week, hours_from_start
    'num_historical_categorical': 1,        # 
    'num_static_numeric': 0,
    'num_static_categorical': 1,
    'num_future_numeric': 0,
    'num_future_categorical': 0,
    'historical_categorical_cardinalities': [],
    'static_categorical_cardinalities': [370],
    'future_categorical_cardinalities': [],
}

# Add the input structure to the configuration
configuration['data_props'] = structure

# Create the model with the configuration.
model = TemporalFusionTransformer(config=OmegaConf.create(configuration), debug=debug, tag=tag, use_mycaffe=use_mycaffe, path=path, lstm_use_mycaffe=lstm_use_mycaffe, use_mycaffe_model=use_mycaffe_model)

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
    elif isinstance(m, nn.Linear) or isinstance(m, LinearEx):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM) or isinstance(m, LstmEx):
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

# Set the devie to CUDA if available
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

model.to(device)
save_weights(model)

# Initialize the optimizer and point it to the model parameters.
if use_mycaffe:
    opt = AdamW2(filter(lambda p: p.requires_grad, list(model.parameters())),
                    lr=configuration['optimization']['learning_rate'],
                    weight_decay=0)
else:
    opt = optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                    lr=configuration['optimization']['learning_rate'])

#========================================================================
# Data Preparation
#========================================================================

# Data preparation - The DictDataSet digests the dictionary of numpy arrays, and output
# as a dictionary of tensors with keys corresponding to the original input dictionary.
# This class will be wrapped with a dedicated dataloader object which creates batches
# of dict objects.
class DictDataSet(Dataset):
    def __init__(self, array_dict: Dict[str, np.ndarray]):
        self.keys_list = []
        for k, v in array_dict.items():
            self.keys_list.append(k)
            if np.issubdtype(v.dtype, np.dtype('bool')):
                setattr(self, k, torch.ByteTensor(v))
            elif np.issubdtype(v.dtype, np.int8):
                setattr(self, k, torch.CharTensor(v))
            elif np.issubdtype(v.dtype, np.int16):
                setattr(self, k, torch.ShortTensor(v))
            elif np.issubdtype(v.dtype, np.int32):
                setattr(self, k, torch.IntTensor(v))
            elif np.issubdtype(v.dtype, np.int64):
                setattr(self, k, torch.LongTensor(v))
            elif np.issubdtype(v.dtype, np.float32):
                setattr(self, k, torch.FloatTensor(v))
            elif np.issubdtype(v.dtype, np.float64):
                setattr(self, k, torch.DoubleTensor(v))
            else:
                setattr(self, k, torch.FloatTensor(self.convertToFloat(v)))

    def convertToFloat(self, v):
        t = []
        for v1 in v:
            dtval = v1.asm8.astype('datetime64[s]').astype(int)
            t.append(dtval)
        t = np.array(t)
        return t.astype(np.float32)

    def __getitem__(self, index):
        return {k: getattr(self, k)[index] for k in self.keys_list}

    def __len__(self):
        return getattr(self, self.keys_list[0]).shape[0]

# Used to create infinite data loader that continually loads data on next.
def recycle(iterable):
    while True:
        for x in iterable:
            yield x

# Converts the dict into a DictDataSet object and creates two dataloaders for each set.
def get_set_and_loaders(data_dict: Dict[str, np.ndarray],
                        shuffled_loader_config: Dict,
                        serial_loader_config: Dict,
                        ignore_keys: List[str] = None,
                        ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset = DictDataSet({k:v for k,v in data_dict.items() if (ignore_keys and k not in ignore_keys)})
    loader = torch.utils.data.DataLoader(dataset,**shuffled_loader_config)
    serial_loader = torch.utils.data.DataLoader(dataset,**serial_loader_config)

    return dataset,iter(recycle(loader)),serial_loader

# We set the configuration for the shuffled data loaders, and for the serial ones and also set the 
# meta_keys which specifies which keys do not contain actual data.
shuffled_loader_config = {'batch_size': configuration['optimization']['batch_size']['training'],
                'drop_last': True,
                'shuffle':False} #True

serial_loader_config = {'batch_size': configuration['optimization']['batch_size']['inference'],
                'drop_last': False,
                'shuffle':False}

# the following fields do not contain actual data, but are only identifiers of each observation
meta_keys = ['combination_id'] # ['time_index','combination_id']

# We use the utility functions for generating the required data loaders for each of the subsets.
train_set,train_loader,train_serial_loader = get_set_and_loaders(data['data_sets']['train'],
                                                                shuffled_loader_config,
                                                                serial_loader_config,
                                                                ignore_keys=meta_keys)
validation_set,validation_loader,validation_serial_loader = get_set_and_loaders(data['data_sets']['validation'],
                                                                shuffled_loader_config,
                                                                serial_loader_config,
                                                                ignore_keys=meta_keys)
test_set,test_loader,test_serial_loader = get_set_and_loaders(data['data_sets']['test'],
                                                                shuffled_loader_config,
                                                                serial_loader_config,
                                                                ignore_keys=meta_keys)

#========================================================================
# Training Procedure
#========================================================================

# Training Procedure - The QueueAggregator helps iwth orchestration of the training process, by
# operating as a running-window aggregator of the training performance metric. This is used for 
# smoothing out the loss during training.
class QueueAggregator(object):
    def __init__(self, max_size):
        self._queued_list = []
        self.max_size = max_size

    def append(self, elem):
        self._queued_list.append(elem)
        if len(self._queued_list) > self.max_size:
            self._queued_list.pop(0)

    def get(self):
        return self._queued_list

# EarlyStopping is used to monitor the performance of the valication set, and indicates when we can
# quit training.
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

# Parameters that contol the training:
# If early stopping is not triggered, after how many epochs should we quit training
max_epochs = 10000
# how many training batches will compose a single training epoch
epoch_iters = 200
# upon completing a training epoch, we perform an evaluation of all the subsets
# eval_iters will define how many batches of each set will compose a single evaluation round
eval_iters = 500
# during training, on what frequency should we display the monitored performance
log_interval = 20
# what is the running-window used by our QueueAggregator object for monitoring the training performance
ma_queue_size = 50
# how many evaluation rounds should we allow,
# without any improvement in the performance observed on the validation set
patience = 5

# initialize early stopping mechanism
es = EarlyStopping(patience=patience)
# initialize the loss aggregator for running window performance estimation
loss_aggregator = QueueAggregator(max_size=ma_queue_size)

# initialize counters
batch_idx = 0
epoch_idx = 0

# For computing loss, we are seeking to optimize corresponding to the actual quantiles we want to estimate.
quantiles_tensor = torch.tensor(configuration['model']['output_quantiles']).to(device)

def calculate_accuracy(predicted_quantiles, labels):
    nNum = labels.shape[0]
    nSteps = labels.shape[1]
    nTotal = 0
    nCorrect = 0

    for i in range(nNum):
        for j in range(nSteps):
            nTotal = nTotal + 1

            fUpper = predicted_quantiles[i][j][0]
            fMid = predicted_quantiles[i][j][1]
            fLower = predicted_quantiles[i][j][2]
            fActual = labels[i][j]

            fUpperRange = abs(fUpper - fMid)
            fLowerRange = abs(fMid - fLower)
            fUpperTarget = fMid + fUpperRange * 0.2
            fLowerTarget = fMid - fLowerRange * 0.2

            if fActual >= fLowerTarget and fActual <= fUpperTarget:
                nCorrect = nCorrect + 1

    fAccuracy = nCorrect / nTotal
    print("Accuracy = %lf %%" % (fAccuracy * 100))
    return fAccuracy

# The batch processing transfers each batch component to the device, feeds the batch to the model,
# computes the loss using: the labels output, and the fixed tensor quantiles tensor that we want to estimate.
def process_batch(idx, batch: Dict[str,torch.tensor],
                  model: nn.Module,
                  quantiles_tensor: torch.tensor,
                  device:torch.device):
    if is_cuda:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

    predicted_quantiles = None

    if use_mycaffe_model_direct:
        q_loss, q_risk = model(batch)        
        if (len(q_loss) > 1):
            q_risk = q_loss[1]
        if (len(q_loss) > 2):
            predicted_quantiles = q_loss[2]
        if (len(q_loss) > 3):
            labels = q_loss[3]
        if (len(q_loss) > 1):
            q_loss = q_loss[0]
    elif use_mycaffe_model:
        q_loss, q_risk = model(batch)        
    else:
        batch_outputs = model(batch)
        labels = batch['target']

        predicted_quantiles = batch_outputs['predicted_quantiles']
        q_loss, q_risk, _ = tft_loss.get_quantiles_loss_and_q_risk(outputs=predicted_quantiles,
                                                                  targets=labels,
                                                                  desired_quantiles=quantiles_tensor, debug_opt=debug, path=path)
    if idx == 0 and predicted_quantiles != None:
        calculate_accuracy(predicted_quantiles, labels)        

    return q_loss, q_risk


#========================================================================
# Training Loop
#========================================================================

def save_batch(i, batch):
    strPath = "test/" + path + "/iter_%d/data/" % i
    DebugFunction.trace_ex(strPath, batch['static_feats_numeric'], "%d_static_feats_numeric" % i)
    DebugFunction.trace_ex(strPath, batch['static_feats_categorical'], "%d_static_feats_categorical" % i)
    DebugFunction.trace_ex(strPath, batch['historical_ts_numeric'], "%d_historical_ts_numeric" % i)
    DebugFunction.trace_ex(strPath, batch['historical_ts_categorical'], "%d_historical_ts_categorical" % i)
    DebugFunction.trace_ex(strPath, batch['future_ts_numeric'], "%d_future_ts_numeric" % i)
    DebugFunction.trace_ex(strPath, batch['future_ts_categorical'], "%d_future_ts_categorical" % i)
    DebugFunction.trace_ex(strPath, batch['target'], "%d_target" % i)
    DebugFunction.trace_ex(strPath, batch['time_index'], "%d_time_index" % i)
    
def load_batch(i, batch):
    strPath = "test/" + path + "/iter_%d/data/" % i
    batch['static_feats_numeric'] = torch.from_numpy(np.load(strPath + "%d_static_feats_numeric.npy" % i))
    batch['static_feats_categorical'] = torch.from_numpy(np.load(strPath + "%d_static_feats_categorical.npy" % i))
    batch['historical_ts_numeric'] = torch.from_numpy(np.load(strPath + "%d_historical_ts_numeric.npy" % i))
    batch['historical_ts_categorical'] = torch.from_numpy(np.load(strPath + "%d_historical_ts_categorical.npy" % i))
    batch['future_ts_numeric'] = torch.from_numpy(np.load(strPath + "%d_future_ts_numeric.npy" % i))
    batch['future_ts_categorical'] = torch.from_numpy(np.load(strPath + "%d_future_ts_categorical.npy" % i))
    batch['target'] = torch.from_numpy(np.load(strPath + "%d_target.npy" % i))
    batch['time_index'] = torch.from_numpy(np.load(strPath + "%d_time_index.npy" % i))

# Training Loop - this loop continues until max_epoch is hit or EarlyStopping is triggered.
# Each epoch starts with the evaluation of each of the subsets by processing eval_iters batches
# and computing the loss which is then averaged.
# After completing the evaluation, a training round is initiated. For each training batch, the
# computed loss is used for calling the optimizer to update the model weights and the loss aggregator.
validation_loss = None

while epoch_idx < max_epochs:
    print(f"Starting Epoch Index {epoch_idx}")

    # evaluation round
    if test:
        model.eval()
        with torch.no_grad():
            # for each subset
            for subset_name, subset_loader in zip(['train','validation','test'],[train_loader,validation_loader,test_loader]):
                print(f"Evaluating {subset_name} set")

                q_loss_vals, q_risk_vals = [],[] # used for aggregating performance along the evaluation round
                for i in range(eval_iters):
                    # get batch
                    batch = next(subset_loader)
                    # process batch
                    batch_loss,batch_q_risk = process_batch(idx=i,batch=batch,model=model,quantiles_tensor=quantiles_tensor,device=device)
                    # accumulate performance
                    q_loss_vals.append(batch_loss)
                    q_risk_vals.append(batch_q_risk)

                # aggregate and average
                eval_loss = torch.stack(q_loss_vals).mean(axis=0)
                eval_q_risk = torch.stack(q_risk_vals,axis=0).mean(axis=0)

                # keep for feeding the early stopping mechanism
                if subset_name == 'validation':
                    validation_loss = eval_loss

                # log performance
                print(f"Epoch: {epoch_idx}, Batch Index: {batch_idx}" + \
                      f"- Eval {subset_name} - " + \
                      f"q_loss = {eval_loss:.5f} , " + \
                      " , ".join([f"q_risk_{q:.1} = {risk:.5f}" for q,risk in zip(quantiles_tensor,eval_q_risk)]))

    # switch to training mode
    model.train()

    # update early stopping mechanism and stop if triggered
    if validation_loss != None:
        if es.step(validation_loss):
            print('Performing early stopping...!')
            break

    # initiating a training round
    for i in range(epoch_iters):
        start_iter_time = time.time()
        if debug:
            DebugFunction.set_output_path(path, i)

        # get training batch
        batch = next(train_loader)

        if use_mycaffe_data:
            rgData = mycaffe.model_fwd_start()
            x_num_static = rgData[0]
            x_cat_static = rgData[1]
            x_num_past = rgData[2]
            x_cat_past = rgData[3]
            x_num_future = rgData[4]
            x_cat_future = rgData[5]
            targets = rgData[6]

            batch['static_feats_categorical'] = torch.from_numpy(x_cat_static).to(device).to(int)
            batch['historical_ts_numeric'] = torch.from_numpy(x_num_past).to(device)
            batch['historical_ts_categorical'] = torch.from_numpy(x_cat_past).to(device).to(int)
            batch['future_ts_numeric'] = torch.from_numpy(x_num_future).to(device)
            batch['future_ts_categorical'] = torch.from_numpy(x_cat_future).to(device).to(int)
            batch['target'] = torch.from_numpy(targets).to(device)

        if debug == True:
            #load_batch(i, batch)
            save_batch(i, batch)

            for param in model.state_dict():
                DebugFunction.trace(model.state_dict()[param], "tft.all." + param, "weights")     

        opt.zero_grad()
        # process batch
        loss,_ = process_batch(idx=i, batch=batch,
                              model=model,
                              quantiles_tensor=quantiles_tensor,
                              device=device)
        if debug:
            model.past_lstm.save_wts("tft.all.", "weights")
            model.future_lstm.save_wts("tft.all.", "weights")

        # compute gradients
        if use_mycaffe_model_direct:
            model.backward_direct()
        else:
            loss.backward()

        if debug==True:
            #model.past_lstm.save_grad("tft.all.", "weights")
            #model.future_lstm.save_grad("tft.all.", "weights")
            DebugFunction.trace(loss, "tft.all.loss")

        # gradient clipping
        if configuration['optimization']['max_grad_norm'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), configuration['optimization']['max_grad_norm'])
        # update weights

        if use_mycaffe_model_direct == False:
            opt.step()
            if use_mycaffe and use_mycaffe_model == False:
                model.past_lstm.update_wts(0.001, 0, 0.9, 0.999, i+1, 1e-08)
                model.future_lstm.update_wts(0.001, 0, 0.9, 0.999, i+1, 1e-08)

        if use_mycaffe or use_mycaffe_model or use_mycaffe_model_direct:
            model.update(i)

        # accumulate performance
        loss_aggregator.append(loss.item())

        # log performance
        #if batch_idx % log_interval == 0:
        print(f"Epoch: {epoch_idx}, Batch Index: {batch_idx} - Train Loss = {np.mean(loss_aggregator.get())}")

        # completed batch
        batch_idx += 1

        end_iter_time = time.time()
        print("Iter Time = %lf seconds" % (end_iter_time - start_iter_time))

    # completed epoch
    epoch_idx += 1


#========================================================================
# Explore Model Outputs
#========================================================================

import tft_torch.visualize as tft_vis

# Apply model by running the inference on the validation set.
model.eval() # switch to evaluation mode

output_aggregator = dict() # will be used for aggregating the outputs across batches

with torch.no_grad():
    # go over the batches of the serial data loader
    for batch in tqdm(validation_serial_loader):
        # process each batch
        if is_cuda:
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device)
        batch_outputs = model(batch)

        # accumulate outputs, as well as labels
        for output_key,output_tensor in batch_outputs.items():
            output_aggregator.setdefault(output_key,[]).append(output_tensor.cpu().numpy())
        output_aggregator.setdefault('target',[]).append(batch['target'].cpu().numpy())

# Stack the outputs form all batches
validation_outputs = dict()
for k in list(output_aggregator.keys()):
    validation_outputs[k] = np.concatenate(output_aggregator[k],axis=0)

# Arbitrary sample index for demonstration
chosen_idx = 42421

#+++ Target Signal Trajectory +++
# Inputs are scaled, and this first visualization shows the 'scaled' output.

# the name of the target signal
target_signal = 'log_sales'
# its relative index among the set of historical numeric input variables
target_var_index = feature_map['historical_ts_numeric'].index(target_signal)
# the quantiles estimated by the trained model
model_quantiles = configuration['model']['output_quantiles']

tft_vis.display_target_trajectory(signal_history=data['data_sets']['validation']['historical_ts_numeric'][...,target_var_index],
                                  signal_future=validation_outputs['target'],
                                  model_preds=validation_outputs['predicted_quantiles'],
                                  observation_index=chosen_idx,
                                  model_quantiles=model_quantiles,
                                  unit='Days')

# In some cases we would like to observe the actual scale of the target variable.
# The scale_back function transforms the values back to their unscaled values.
def scale_back(scaler_obj,signal):
    inv_trans = scaler_obj.inverse_transform(copy.deepcopy(signal))
    return np.power(10,inv_trans) - 1
transform_back = partial(scale_back,data['scalers']['numeric'][target_signal])

tft_vis.display_target_trajectory(signal_history=data['data_sets']['validation']['historical_ts_numeric'][...,target_var_index],
                                  signal_future=validation_outputs['target'],
                                  model_preds=validation_outputs['predicted_quantiles'],
                                  observation_index=chosen_idx,
                                  model_quantiles=model_quantiles,
                                  unit='Days',
                                  transformation=transform_back)

#========================================================================
# Selection Weights
#========================================================================

# The TFT has an internal way of selecting variables.  Each input channel has a separate dedicated
# method, historical temporal data, static descriptors data, known future inputs data, etc.
static_feats = feature_map['static_feats_numeric'] + feature_map['static_feats_categorical']
historical_feats = feature_map['historical_ts_numeric'] + feature_map['historical_ts_categorical']
future_feats = feature_map['future_ts_numeric'] + feature_map['future_ts_categorical']

# the precentiles to compute for describing the distribution of the weights
weights_prctile = [10,50,90]

# Show the aggregation and ordering of the attributes for each input channel separately (show
# the specified percentiles of the weights distribution for each feature).
mapping = {
    'Static Weights': {'arr_key': 'static_weights', 'feat_names':static_feats},
    'Historical Weights': {'arr_key': 'historical_selection_weights', 'feat_names':historical_feats},
    'Future Weights': {'arr_key': 'future_selection_weights', 'feat_names':future_feats},
}
tft_vis.display_selection_weights_stats(outputs_dict=validation_outputs,
                                       prctiles=weights_prctile,
                                       mapping=mapping,
                                       sort_by=50)

# Graphically show the selection weights.
# static attributes
tft_vis.display_sample_wise_selection_stats(weights_arr=validation_outputs['static_weights'],
                                           observation_index=chosen_idx,
                                           feature_names=static_feats,
                                           top_n=20,
                                           title='Static Features')

# historical temporal attributes
tft_vis.display_sample_wise_selection_stats(weights_arr=validation_outputs['historical_selection_weights'],
                                           observation_index=chosen_idx,
                                           feature_names=historical_feats,
                                           top_n=20,
                                           title='Historical Features',
                                           rank_stepwise=True)

# futuristic (known) temporal attributes
tft_vis.display_sample_wise_selection_stats(weights_arr=validation_outputs['future_selection_weights'],
                                           observation_index=chosen_idx,
                                           feature_names=future_feats,
                                           top_n=20,
                                           title='Future Features',
                                           historical=False,
                                           rank_stepwise=False)

#========================================================================
# Attention Scores
#========================================================================

# TFT uses internal attention to weight the information from the sequential data.  The
# attention scores are used to infer which preceding time-steps affected the output.

tft_vis.display_attention_scores(attention_scores=validation_outputs['attention_scores'],
                                horizons=1,
                                prctiles=[10,50,90],
                                unit='Days')

# Display multi-horizon attention scores
tft_vis.display_attention_scores(attention_scores=validation_outputs['attention_scores'],
                                horizons=[1,3,5],
                                prctiles=50,
                                unit='Days')

# Display the attention scores associated with each output horizon.
tft_vis.display_sample_wise_attention_scores(attention_scores=validation_outputs['attention_scores'],
                                            observation_index=chosen_idx,
                                            horizons=[1,5,10],
                                            unit='Days')

print('done!')