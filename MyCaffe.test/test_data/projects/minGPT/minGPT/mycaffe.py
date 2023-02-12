from curses.ascii import isalnum
from constants import *
import clr
import os
import numpy as np
import System
clr.AddReference("C:\\temp\\projects\\2023.minGpt\\minGPT\\MyCaffeConnector\\bin\\Debug\\MyCaffeConnector.dll")
from MyCaffeConnector import *
from System import Array, Single
import ctypes

from System.Runtime.InteropServices import GCHandle, GCHandleType

class MyCaffe():
    def __init__(self, bFullInit):
        self.innerprod_shapes = {}
        pid = os.getpid()
        print("OS PID = {%d}" % (pid))
        self.mycaffe = MyCaffeConnector()
        if bFullInit:
            self.mycaffe.InitializeEx(batch_size, seq_len, d_model, sp_vocab_size, sp_vocab_size, 0.0)
        else:
            self.mycaffe.Initialize()
        print("Loaded MyCaffe.")
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def test(self, val):
        self.mycaffe.Test(list(val.data))

    def getname(self, tag):
        name1 = ""
        for char in tag:
            if char.isdigit():
                name1 += char
            elif char.isalnum():
                name1 += char
        return name1

    def tfb_fwd(self, tag, x):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2]
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.tfb_fwd(tag, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def tfb_bwd(self, tag, y, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2]
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.tfb_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(rgShape)
        xgrad = xgrad.to(device)
        del rgOut
        del out
        return xgrad

    def tfb_fwd_all(self, tag, x):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2]
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.tfb_fwd_all(tag, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def tfb_bwd_all(self, tag, y, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2]
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.tfb_bwd_all(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(rgShape)
        xgrad = xgrad.to(device)
        del rgOut
        del out
        return xgrad


    def softmax_fwd(self, tag, x):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2]
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.softmax_fwd(tag, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def softmax_bwd(self, tag, y, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2]
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.softmax_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(rgShape)
        xgrad = xgrad.to(device)
        del rgOut
        del out
        return xgrad

    def logsoftmax_fwd(self, tag, x):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2]
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.logsoftmax_fwd(tag, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def logsoftmax_bwd(self, tag, y, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2]
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.logsoftmax_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(rgShape)
        xgrad = xgrad.to(device)
        del rgOut
        del out
        return xgrad
    
    def layernorm_fwd(self, tag, x):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2]
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.layernorm_fwd(tag, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def layernorm_bwd(self, tag, y, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2]
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.layernorm_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(rgShape)
        xgrad = xgrad.to(device)
        del rgOut
        del out
        return xgrad

    def innerproduct_setup(self, tag, nAxis, infeat, outnum, wt, bias):
        if bias != None:
            bias = list(bias.detach().cpu().numpy().flatten().data)
        self.innerprod_shapes[tag] = (nAxis, infeat, outnum)
        self.mycaffe.innerproduct_setup(tag, nAxis, batch_size, seq_len, infeat, outnum, list(wt.detach().cpu().numpy().flatten().data), bias)
        
    def innerproduct_fwd(self, tag, x):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2]
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.innerproduct_fwd(tag, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        nAxis = self.innerprod_shapes[tag][0]
        outNum = self.innerprod_shapes[tag][2]
        rgShape2 = [nN, nC, nH]
        if nW > 1 or nAxis == 3:
            rgShape2.append(nW)
        rgShape2[nAxis] = outNum
        y = y.reshape(rgShape2)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def innerproduct_bwd(self, tag, y, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2]
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.innerproduct_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        nAxis = self.innerprod_shapes[tag][0]
        inNum = self.innerprod_shapes[tag][1]
        rgShape2 = [nN, nC, nH]
        if nW > 1 or nAxis == 3:
            rgShape2.append(nW)
        rgShape2[nAxis] = inNum
        xgrad = xgrad.reshape(rgShape2)
        xgrad = xgrad.to(device)
        del rgOut
        del out
        return xgrad
    
    def sum(self, x, axis, keepdims):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2]
        rgOut = self.mycaffe.channel_sum(nN, nC, nH, list(x.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y

    def step(self, iter, enc, trg):
        rgOut = self.mycaffe.Step(iter, list(enc.detach().cpu().numpy().flatten().data), list(trg.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)        
        tensorOutput = torch.from_numpy(out).float()
        tensorOutput = tensorOutput.reshape(batch_size, seq_len, sp_vocab_size);
        del out
        
        return tensorOutput
    
    def current_loss(self):
        return self.mycaffe.CurrentLoss

    def current_accuracy(self):
        return self.mycaffe.CurrentAccuracy

_MAP_NP_NET = {
    np.dtype('float32'): System.Single,
    np.dtype('float64'): System.Double,
    np.dtype('int8')   : System.SByte,
    np.dtype('int16')  : System.Int16,
    np.dtype('int32')  : System.Int32,
    np.dtype('int64')  : System.Int64,
    np.dtype('uint8')  : System.Byte,
    np.dtype('uint16') : System.UInt16,
    np.dtype('uint32') : System.UInt32,
    np.dtype('uint64') : System.UInt64,
    np.dtype('bool')   : System.Boolean,
}
_MAP_NET_NP = {
    'Single' : np.dtype('float32'),
    'Double' : np.dtype('float64'),
    'SByte'  : np.dtype('int8'),
    'Int16'  : np.dtype('int16'), 
    'Int32'  : np.dtype('int32'),
    'Int64'  : np.dtype('int64'),
    'Byte'   : np.dtype('uint8'),
    'UInt16' : np.dtype('uint16'),
    'UInt32' : np.dtype('uint32'),
    'UInt64' : np.dtype('uint64'),
    'Boolean': np.dtype('bool'),
}

def asNumpyArray(netArray):
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    '''
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    netType = netArray.GetType().GetElementType().Name

    try:
        npArray = np.empty(dims, order='C', dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType) )

    try: # Memmove 
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__['data'][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated: sourceHandle.Free()
    return npArray

