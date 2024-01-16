import os
from pickle import NONE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cwd = os.getcwd()

from typing import Dict,List,Tuple
import torch
from torch import optim
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from omegaconf import OmegaConf,DictConfig
import time
from datetime import datetime
from tqdm import tqdm

from mom_trans.model_inputs import ModelFeatures
from settings.default import QUANDL_TICKERS
from tft_torch.tft import TemporalFusionTransformer, LinearEx, LstmEx, mycaffe
from utility import DebugFunction, save_weights, save_batch, save_loss, save_weights_ex

torch.manual_seed(42)
torch.cuda.seed_all()

os.chdir(cwd)
print (os.getcwd())

debug = True
use_mycaffe = True
use_mycaffe_data = False
use_mycaffe_model_direct = False
use_mycaffe_model = False
lstm_use_mycaffe = True
linear_use_mycaffe = True
matmul_use_mycaffe = True
clone_use_mycaffe = True
save_data = False
tag = "tft.all"
test = False
path = "all"
strName = "tft.sharpe"

ASSET_CLASS_MAPPING = dict(zip(QUANDL_TICKERS, ["COMB"] * len(QUANDL_TICKERS)))

total_time_steps = 63
train_interval0 = 1990
train_interval1 = 2019
train_interval2 = 2020
changepoint_lbws = None
split_tickers_individually = True
train_valid_ratio = 0.9
architecture="TFT"
time_features=False
force_output_sharpe_length=None

asset_class_dictionary = ASSET_CLASS_MAPPING
features_file_path = 'data\quandl_cpd_nonelbw.csv'
raw_data = pd.read_csv(features_file_path, index_col=0, parse_dates=True)
raw_data["date"] = raw_data["date"].astype("datetime64[ns]")

# We have some configuration settings to make related to the optimization methodology and model structure.
configuration = {'optimization':
                 {
                     'batch_size': {'training': 256, 'inference': 256},
                     'learning_rate': 0.01,
                     'max_grad_norm': 0, #1.0,
                 }
                 ,
                 'model':
                 {
                     'dropout': 0,
                     'state_size': 10,
                     'output_quantiles': [0.1, 0.5, 0.9],
                     'lstm_layers': 2,
                     'attention_heads': 4
                 },
                 # these arguments are related to possible extensions of the model class
                 #'task_type':'regression',
                 'task_type':'classification',
                 'target_window_start': None
                }

# TODO more/less than the one year test buffer
model_features = ModelFeatures(
    raw_data,
    total_time_steps,
    start_boundary=train_interval0,
    test_boundary=train_interval1,
    test_end=train_interval2,
    changepoint_lbws=changepoint_lbws,
    split_tickers_individually=split_tickers_individually,
    train_valid_ratio=train_valid_ratio,
    add_ticker_as_static=(architecture == "TFT"),
    time_features=time_features,
    lags=force_output_sharpe_length,
    asset_class_dictionary=asset_class_dictionary,
)

class ModelDataSet(Dataset):
    def __init__(self, raw_data: dict):
        self.data = { 
            'static_feats_categorical' : raw_data['inputs'][:,-1:,8:].astype(int),
            'historical_ts_numeric': raw_data['inputs'][:,:,0:8].astype(np.float32),
            'target': raw_data['outputs'].astype(np.float32) }
        self.keys_list = []
        for k, v in self.data.items():
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

    def __len__(self):
        return getattr(self, self.keys_list[0]).shape[0]

    def __getitem__(self, idx):
        batch = {k: getattr(self, k)[idx] for k in self.keys_list}
        batch['static_feats_categorical'] = batch['static_feats_categorical'].reshape(1)
        return batch

# Used to create infinite data loader that continually loads data on next.
def recycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_set_and_loaders(raw_data)-> Tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset = ModelDataSet(raw_data)
    loader = DataLoader(dataset, batch_size=configuration['optimization']['batch_size']['training'], shuffle=False)
    return dataset, iter(recycle(loader))

train_set, train_loader = get_set_and_loaders(model_features.train)
valid_set, valid_loader = get_set_and_loaders(model_features.valid)
test_set, test_loader = get_set_and_loaders(model_features.test_sliding)

structure = {
    'num_historical_numeric': 8,           
    'num_historical_categorical': 0,        
    'num_static_numeric': 0,
    'num_static_categorical': 1,
    'num_future_numeric': 0,
    'num_future_categorical': 0,
    'historical_categorical_cardinalities': [],
    'static_categorical_cardinalities': [512],
    'future_categorical_cardinalities': [],
}

# Add the input structure to the configuration
configuration['data_props'] = structure

model = None
opt = None

if use_mycaffe_model_direct == False:
    model = TemporalFusionTransformer(decoder_only=True,config=OmegaConf.create(configuration), debug=debug, tag=tag, use_mycaffe=use_mycaffe, path=path, lstm_use_mycaffe=lstm_use_mycaffe, linear_use_mycaffe=linear_use_mycaffe, matmul_use_mycaffe=matmul_use_mycaffe, clone_use_mycaffe=clone_use_mycaffe, use_mycaffe_model=use_mycaffe_model)

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

if use_mycaffe_model_direct == False:
    model.apply(weight_init)

# Set the devie to CUDA if available
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

if use_mycaffe_model_direct == False:
    model.to(device)
    save_weights(model, strName)
    opt = optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                        lr=configuration['optimization']['learning_rate'])

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

activation = nn.Tanh()

def sharpe_loss(weights, y_true):
        debug = DebugFunction.apply  
        
        if debug:
            DebugFunction.trace(weights, "sharpe.weights")
            weights = debug(weights)

        if debug:        
            DebugFunction.trace(y_true, "sharpe.y_true")
            y_true = debug(y_true)
    
        captured_returns = weights * y_true         

        if debug:        
            DebugFunction.trace(captured_returns, "sharpe.captured_returns")
            captured_returns = debug(captured_returns)

        #sum_returns = torch.sum(captured_returns)
             
        mean_returns = torch.mean(captured_returns)

        if debug:
            DebugFunction.trace(mean_returns, "sharpe.mean_returns")
            mean_returns = debug(mean_returns)

        mean_returns_sq = torch.square(mean_returns)

        if debug:
            DebugFunction.trace(mean_returns_sq, "sharpe.mean_returns_sq")
            mean_returns_sq = debug(mean_returns_sq)        

        captured_returns_sq = torch.square(captured_returns)

        if debug:        
            DebugFunction.trace(captured_returns_sq, "sharpe.captured_returns_sq")
            captured_returns_sq = debug(captured_returns_sq)        

        mean_captured_returns_sq = torch.mean(captured_returns_sq)

        if debug:   
            DebugFunction.trace(mean_captured_returns_sq, "sharpe.mean_captured_returns_sq")
            mean_captured_returns_sq = debug(mean_captured_returns_sq)        

        mean_captured_returns_sq_minus_mean_returns_sq = mean_captured_returns_sq - mean_returns_sq

        if debug:
            DebugFunction.trace(mean_captured_returns_sq_minus_mean_returns_sq, "sharpe.mean_captured_returns_sq_minus_mean_returns_sq")
            mean_captured_returns_sq_minus_mean_returns_sq = debug(mean_captured_returns_sq_minus_mean_returns_sq)        

        twofiftytwo = torch.tensor(252.0)

        mean_captured_returns_sq_minus_mean_returns_sqrt = torch.sqrt(mean_captured_returns_sq_minus_mean_returns_sq) 

        if debug:   
            DebugFunction.trace(mean_captured_returns_sq_minus_mean_returns_sqrt, "sharpe.mean_captured_returns_sq_minus_mean_returns_sqrt")
            mean_captured_returns_sq_minus_mean_returns_sqrt = debug(mean_captured_returns_sq_minus_mean_returns_sqrt)        

        loss1 = (mean_returns / mean_captured_returns_sq_minus_mean_returns_sqrt)

        if debug:        
            DebugFunction.trace(loss1, "sharpe.loss1")
            loss1 = debug(loss1)        

        loss2 = loss1 * torch.sqrt(twofiftytwo)

        if debug:        
            DebugFunction.trace(loss2, "sharpe.loss2")
            loss2 = debug(loss2)        

        loss = loss2 * -1

        if debug:   
            DebugFunction.trace(loss, "sharpe.loss")
            loss = debug(loss)        
        
        return loss

# The batch processing transfers each batch component to the device, feeds the batch to the model,
# computes the loss using: the labels output, and the fixed tensor quantiles tensor that we want to estimate.
def process_batch(idx, batch: Dict[str,torch.tensor],
                  model: nn.Module,
                  device:torch.device):
    if is_cuda:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

    predicted_pos = None
    
    if debug:
        save_batch(idx, strName, batch)

    if use_mycaffe_model_direct == False:
        batch_outputs = model(batch)
        labels = batch['target']

        predicted_pos = batch_outputs['predicted_quantiles']
        
        DebugFunction.trace(predicted_pos, "predicted_pos")
        predicted_pos = DebugFunction.apply(predicted_pos)

        if use_mycaffe_model == False:
            s_loss = sharpe_loss(predicted_pos, labels)
        else:
            s_loss = model.forward_direct(predicted_pos, labels)

        if debug:
            DebugFunction.trace(predicted_pos, "predicted_pos")
            DebugFunction.trace(s_loss, "s_loss")
    else:
        batch_outputs = mycaffe.model_fwd(batch["static_feats_categorical"], batch["historical_ts_numeric"], batch["target"])    
        s_loss = batch_outputs["loss"]
        
    return s_loss

#========================================================================
# Training Loop
#========================================================================
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
            for subset_name, subset_loader in zip(['train','validation','test'],[train_loader,valid_loader,test_loader]):
                print(f"Evaluating {subset_name} set")

                s_loss_vals = [] # used for aggregating performance along the evaluation round
                for i in tqdm(range(eval_iters),desc="testing"):
                    # get batch
                    batch = next(subset_loader)
                    # process batch
                    batch_loss = process_batch(idx=i,batch=batch,model=model,device=device)
                    # accumulate performance
                    s_loss_vals.append(batch_loss)

                # aggregate and average
                eval_loss = torch.stack(s_loss_vals).mean(axis=0)

                # keep for feeding the early stopping mechanism
                if subset_name == 'validation':
                    validation_loss = eval_loss

                # log performance
                print(f"Epoch: {epoch_idx}, Batch Index: {batch_idx}" + \
                      f"- Eval {subset_name} - " + \
                      f"s_loss = {eval_loss:.5f} ")

    # switch to training mode
    if model != None:
        model.train()

    # update early stopping mechanism and stop if triggered
    if validation_loss != None:
        if es.step(validation_loss):
            print('Performing early stopping...!')
            break

    # initiating a training round
    for i in tqdm(range(epoch_iters), desc="training"):
        start_iter_time = time.time()

        # get training batch
        batch = next(train_loader)

        num_samples, num_historical_steps, _ = batch['historical_ts_numeric'].shape
        if num_samples != configuration['optimization']['batch_size']['training']:
            continue
        
        if save_data:
            save_batch(i, 'tft.sharpe.dbg', batch)

        if use_mycaffe == True or use_mycaffe_model == True:
            model.model_clear_diffs()

        if opt != None:
            opt.zero_grad()
        # process batch
        loss = process_batch(idx=i, batch=batch,
                              model=model,
                              device=device)
        print(f"**Pre Backward*** Epoch: {epoch_idx}, Batch Index: {batch_idx} - Train Loss = {loss.item()}")
        if model != None:        
            if save_data:
                save_loss(i, 'tft.sharpe.dbg', loss)
            if debug:
                model.past_lstm.save_wts("", strName + "\\weights\\past_lstm")

            # compute gradients
            loss.backward()

            if debug:
                model.save_grad("tft.sharpe.dbg")

            # gradient clipping
            if configuration['optimization']['max_grad_norm'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), configuration['optimization']['max_grad_norm'])
            # update weights

            opt.step()
        else:
            loss_grad = torch.tensor(1.0)
            mycaffe.model_bwd(loss_grad)    

        if use_mycaffe == True or use_mycaffe_model == True:
            model.update(i)

        if save_data and model != None:
            save_weights_ex(i, model, 'tft.sharpe.dbg')

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

