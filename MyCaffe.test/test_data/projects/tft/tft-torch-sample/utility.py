import os
import numpy as np
import torch
from typing import Dict

input_dict = { None : "" }
loss_weight = 1

def save_blob(strPath, strName, batch : Dict[str,torch.tensor]):
    strFile = strPath + "/" + strName + ".npy";
    data = batch[strName]
    np.save(strFile, data.cpu().detach().numpy())

def save_batch(nIdx, batch):
    strPath = "data/favorita/batch256/batch_%d" % (nIdx)
    save_blob(strPath, "static_feats_numeric", batch)
    save_blob(strPath, "static_feats_categorical", batch)
    save_blob(strPath, "historical_ts_numeric", batch)
    save_blob(strPath, "historical_ts_categorical", batch)
    save_blob(strPath, "future_ts_numeric", batch)
    save_blob(strPath, "future_ts_categorical", batch)
    save_blob(strPath, "target", batch)

def load_blob(strPath, strName, batch : Dict[str, torch.tensor], maxCount):
    strFile = strPath + "/" + strName + ".npy";
    data = torch.from_numpy(np.load(strFile))
    if maxCount != None and maxCount < data.shape[0]:
        data = data[0:maxCount]
    batch[strName] = data

def load_batch(strPath, maxCount = None):
    batch = { }    
    load_blob(strPath, "static_feats_numeric", batch, maxCount)
    load_blob(strPath, "static_feats_categorical", batch, maxCount)
    load_blob(strPath, "historical_ts_numeric", batch, maxCount)
    load_blob(strPath, "historical_ts_categorical", batch, maxCount)
    load_blob(strPath, "future_ts_numeric", batch, maxCount)
    load_blob(strPath, "future_ts_categorical", batch, maxCount)
    load_blob(strPath, "target", batch, maxCount)
    return batch

def save_weights1(model, subPath):    
    strPath = "data/favorita/weights"
    strPath = strPath + "/" + subPath
    idx = 0

    if not os.path.exists(strPath):
        os.makedirs(strPath)

    for param in model.state_dict():
        data = model.state_dict()[param].cpu().detach().numpy()        
        strFile = strPath + "/" + param + ".npy"
        np.save(strFile, data)
        idx = idx + 1

def save_weights(model):
    save_weights1(model.static_transform, "static_transform")
    save_weights1(model.historical_ts_transform, "hist_ts_transform")
    save_weights1(model.future_ts_transform, "future_ts_transform")
    save_weights1(model.static_selection, "static_selection")
    save_weights1(model.historical_ts_selection, "hist_ts_selection")
    save_weights1(model.future_ts_selection, "future_ts_selection")
    save_weights1(model.static_encoder_selection, "static_encoder_selection")
    save_weights1(model.static_encoder_enrichment, "static_encoder_enrichment")
    save_weights1(model.static_encoder_sequential_cell_init, "static_encoder_sequential_cell_init")
    save_weights1(model.static_encoder_sequential_state_init, "static_encoder_sequential_state_init")
    save_weights1(model.past_lstm, "past_lstm")
    save_weights1(model.future_lstm, "future_lstm")
    save_weights1(model.post_lstm_gating, "post_lstm_gating")
    save_weights1(model.static_enrichment_grn, "static_enrichment_grn")
    save_weights1(model.multihead_attn, "multihead_attn")
    save_weights1(model.post_attention_gating, "post_attention_gating")
    save_weights1(model.pos_wise_ff_grn, "pos_wise_ff_grn")
    save_weights1(model.pos_wise_ff_gating, "pos_wise_ff_gating")
    save_weights1(model.output_layer, "output_layer")

class DebugFunction(torch.autograd.Function):
    out_path = "test/"

    @staticmethod
    def set_output_path(path, i):
        if path == "":
            print("DebugFunction missing path")
        DebugFunction.out_path = "test/" + path + "/iter_%d/" % i
        if not os.path.exists(DebugFunction.out_path):
            os.makedirs(DebugFunction.out_path)

    @staticmethod
    def trace(t, name, subpath=None):
        path = DebugFunction.out_path
        if subpath != None:
            path += subpath + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        input_dict.update({t: name})
        filename = path + name + ".npy"
        np.save(filename, t.detach().cpu().numpy())

    @staticmethod
    def trace_ex(strPath, t, name):
        if not os.path.exists(strPath):
            os.makedirs(strPath)
        input_dict.update({t: name})
        filename = strPath + name + ".npy"
        np.save(filename, t.detach().cpu().numpy())

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)       
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        name = input_dict.get(input)
        
        if name == None:
            name = "unknown";

        if name == "tft.all.asp.past_lstm_output":
            print("found it")
            
        if name == "15_loss":
            grad_output = grad_output * loss_weight

        #print("bwd: " + name)
        np.save(DebugFunction.out_path + name + ".grad", grad_output.detach().cpu().numpy())
        return grad_output

