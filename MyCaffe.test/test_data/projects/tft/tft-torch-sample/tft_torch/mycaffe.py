from contextlib import nullcontext
from curses.ascii import isalnum
#from constants import *
import torch
import clr
import os
import numpy as np
import System
clr.AddReference("..\\..\\MyCaffeConnector\\bin\\\Debug\\MyCaffeConnector.dll")
from MyCaffeConnector import *
from System import Array, Single
import ctypes

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

from System.Runtime.InteropServices import GCHandle, GCHandleType

class MyCaffe():
    def __init__(self, bFullInit):
        pid = os.getpid()
        print("OS PID = {%d}" % (pid))
        self.mycaffe = MyCaffeConnector()
        if bFullInit:
            self.mycaffe.InitializeTFT()
        else:
            self.mycaffe.Initialize()
        print("Loaded MyCaffe.")
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.lstm_layers = 2
        self.start = None

    def get_input_data(self, rgX, bGrad):
        rgIn = np.array(len(rgX), dtype=float)
        for i in range(0, len(rgX)):
            x = rgX[i]
            rgIn = np.append(rgIn, len(x.shape))
            for i in range(0, len(x.shape)):
                rgIn = np.append(rgIn, x.shape[i])

        rgrgData = []
        for i in range(0, len(rgX)):
            x = rgX[i]
            rgData = x.detach().cpu().numpy().flatten()
            rgIn = np.append(rgIn, len(rgData))
            rgrgData.append(rgData)

        for i in range(0, len(rgrgData)):
            data = rgrgData[i]
            data = data
            rgIn = np.append(rgIn, data)

        return rgIn

    def get_output_data(self, rgOut):
        if rgOut == None:
            return None
        out = asNumpyArray(rgOut)
        nItems = int(out[0])
        rgCounts = []
        rgData = []
        rgShapes = []

        nOffset = 1
        for i in range(0, nItems):
            rgCounts.append(int(out[nOffset]))
            nOffset += 1

        for i in range(0, nItems):
            nShapeCount = int(out[nOffset])
            nOffset += 1
            rgShape = []
            for j in range(0, nShapeCount):
                rgShape.append(int(out[nOffset]))
                nOffset += 1
            rgShapes.append(rgShape)

        for i in range(0, nItems):
            count = rgCounts[i]
            if count == 0:
                rgData.append(None)
            else:
                data = out[nOffset:nOffset + count]
                data = data.reshape(rgShapes[i])
                rgData.append(data)
            nOffset += count
        return rgData

    def model_fwd_start(self):
        rgOut = self.mycaffe.model_fwd(None, 0, 0)
        return self.get_output_data(rgOut)

    def model_fwd_full(self):
        nStart = 0
        nEnd = -1
        rgIn = self.get_input_data([], False);
        valIn = None
        if len(rgIn.shape) > 0:
            valIn = list(rgIn.data)
        rgOut = self.mycaffe.model_fwd(valIn, nStart, nEnd)
        res = self.get_output_data(rgOut)
        return res

    def model_fwd(self, x1, trg):
        nStart = 28
        self.start = nStart
        nEnd = -1
        rgIn = self.get_input_data([x1, trg], False);
        valIn = None
        if len(rgIn.shape) > 0:
            valIn = list(rgIn.data)
        rgOut = self.mycaffe.model_fwd(valIn, nStart, nEnd)
        return self.get_output_data(rgOut)

    def model_fwd2(self, x1, x2, trg):
        nStart = 23
        self.start = nStart
        nEnd = -1
        rgIn = self.get_input_data([x1, x2, trg], False);
        valIn = None
        if len(rgIn.shape) > 0:
            valIn = list(rgIn.data)
        rgOut = self.mycaffe.model_fwd(valIn, nStart, nEnd)
        return self.get_output_data(rgOut)

    def model_fwd3(self, x1, x2, x3, trg):
        nStart = 23
        self.start = nStart
        nEnd = -1
        rgIn = self.get_input_data([x1, x2, x3, trg], False);
        valIn = None
        if len(rgIn.shape) > 0:
            valIn = list(rgIn.data)
        rgOut = self.mycaffe.model_fwd(valIn, nStart, nEnd)
        return self.get_output_data(rgOut)

    def model_fwd4(self, x1, x2, x3, x4, trg):
        nStart = 17
        self.start = nStart
        nEnd = -1
        rgIn = self.get_input_data([x1, x2, x3, x4, trg], False);
        valIn = None
        if len(rgIn.shape) > 0:
            valIn = list(rgIn.data)
        rgOut = self.mycaffe.model_fwd(valIn, nStart, nEnd)
        return self.get_output_data(rgOut)

    def model_fwd5(self, x1, x2, x3, x4, x5, trg):
        nStart = 17
        self.start = nStart
        nEnd = -1
        rgIn = self.get_input_data([x1, x2, x3, x4, x5, trg], False);
        valIn = None
        if len(rgIn.shape) > 0:
            valIn = list(rgIn.data)
        rgOut = self.mycaffe.model_fwd(valIn, nStart, nEnd)
        return self.get_output_data(rgOut)

    def model_fwd6(self, x1, x2, x3, x4, x5, x6, trg):
        nStart = 10
        self.start = nStart
        nEnd = -1
        rgIn = self.get_input_data([x1, x2, x3, x4, x5, x6, trg], False);
        valIn = None
        if len(rgIn.shape) > 0:
            valIn = list(rgIn.data)
        rgOut = self.mycaffe.model_fwd(valIn, nStart, nEnd)
        return self.get_output_data(rgOut)

    def model_fwd7(self, x1, x2, x3, x4, x5, x6, x7, trg):
        nStart = 11
        self.start = nStart
        nEnd = -1
        rgIn = self.get_input_data([x1, x2, x3, x4, x5, x6, x7, trg], False);
        valIn = None
        if len(rgIn.shape) > 0:
            valIn = list(rgIn.data)
        rgOut = self.mycaffe.model_fwd(valIn, nStart, nEnd)
        return self.get_output_data(rgOut)

    def model_bwd(self, y):
        nStart = self.start if self.start != None else 0
        nEnd = -1
        rgInGrad = self.get_input_data([y], True)
        rgOut = self.mycaffe.model_bwd(list(rgInGrad.data), nStart, nEnd)
        return self.get_output_data(rgOut);

    def model_update(self, nIter):
        self.mycaffe.model_update(nIter)

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

    def sumEx(self, x, axis):
        n = x.shape[0]
        c = x.shape[1] if len(x.shape) > 1 else 1
        h = x.shape[2] if len(x.shape) > 2 else 1
        w = x.shape[3] if len(x.shape) > 3 else 1
        rgIn = self.get_input_data([x], True)
        rgOut = self.mycaffe.sum(n, c, h, w, list(rgIn.data))
        outVal = self.get_output_data(rgOut)
        outVal = torch.from_numpy(outVal[0].reshape((n, c))).to(device)
        return outVal

    def lstm_wts(self, tag):
        rgOut = self.mycaffe.lstm_wts(tag)
        out = asNumpyArray(rgOut)
        nCount = int(out[0])
        data = []
        offset = 1 + nCount
        for i in range(0, nCount):
            count1 = int(out[i + 1])
            data1 = out[offset:offset+count1]
            offset += count1
            data.append(data1)
        return data           

    def lstm_grad(self, tag):
        rgOut = self.mycaffe.lstm_grad(tag)
        out = asNumpyArray(rgOut)
        nCount = int(out[0])
        data = []
        offset = 1 + nCount
        for i in range(0, nCount):
            count1 = int(out[i + 1])
            data1 = out[offset:offset+count1]
            offset += count1
            data.append(data1)
        return data         
    
    def lstm_update_wts(self, tag, lr, decay, beta1, beta2, nT, eps):
        self.mycaffe.lstm_update_wts(tag, lr, decay, beta1, beta2, nT, eps)

    def lstm_fwd(self, tag, x, h, c, nState, nLayers):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2] if len(x.shape) > 2 else 1
        rgOut = self.mycaffe.lstm_fwd(tag, nState, nLayers, nN, nC, nH, list(x.detach().cpu().numpy().flatten().data), list(h.detach().cpu().numpy().flatten().data), list(c.detach().cpu().numpy().flatten().data))        
        out = asNumpyArray(rgOut)
        nData = int(out[0]);
        nHidden = int(out[1]);
        data = out[2:2+nData]
        hidden = out[2+nData:2+nData+nHidden]
        cell = out[2+nData+nHidden:]
        self.lstm_layers = nLayers

        y = torch.from_numpy(data).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        y.requires_grad = True
        h1 = torch.from_numpy(hidden).float()
        h1 = h1.reshape((nLayers, nN, nH))
        h1 = h1.to(device)
        h1.requires_grad = True
        c1 = torch.from_numpy(cell).float()
        c1 = c1.reshape((nLayers, nN, nH))
        c1 = c1.to(device)
        c1.requires_grad = True

        del rgOut
        del out
        del data
        del hidden
        del cell
        return [y, h1, c1]
    
    def lstm_bwd(self, tag, y, h, c, ygrad, hgrad, cgrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2] if len(y.shape) > 2 else 1
        rgH = list(h.detach().cpu().numpy().flatten().data)
        rgC = list(c.detach().cpu().numpy().flatten().data)

        if hgrad[0][0][0].item() != 0 or hgrad[0][0][1].item() != 0 or hgrad[0][0][2].item() != 0 or cgrad[0][0][0].item() != 0 or cgrad[0][0][1].item() != 0 or cgrad[0][0][2].item() != 0:
            rgHdx = list(hgrad.detach().cpu().numpy().flatten().data)
            rgCdx = list(cgrad.detach().cpu().numpy().flatten().data)
        else:
            rgHdx = None
            rgCdx = None

        rgOut = self.mycaffe.lstm_bwd(tag, self.lstm_layers, nN, nC, nH, list(y.detach().cpu().numpy().flatten().data), rgH, rgC, list(ygrad.detach().cpu().numpy().flatten().data), rgHdx, rgCdx)
        out = asNumpyArray(rgOut)

        nData = int(out[0])
        nHidden = int(out[1])
        data = out[2:2+nData]
        hidden = out[2+nData:2+nData+nHidden]
        cell = out[2+nData+nHidden:]

        nLayers = self.lstm_layers
        y = torch.from_numpy(data).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        h = torch.from_numpy(hidden).float()
        h = h.reshape((nLayers, nN, nH))
        h = h.to(device)
        c = torch.from_numpy(cell).float()
        c = c.reshape((nLayers, nN, nH))
        c = c.to(device)

        del rgOut
        del out
        del data
        del hidden
        del cell
        return [y, h, c]

    def softmax_fwd(self, tag, x, nAxis):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2] if len(x.shape) > 2 else 1
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.softmax_fwd(tag, nAxis, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
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
        nH = y.shape[2] if len(y.shape) > 2 else 1
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.softmax_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(rgShape)
        xgrad = xgrad.to(device)
        del rgOut
        del out
        return xgrad

    def elu_fwd(self, tag, x):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2] if len(x.shape) > 2 else 1
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.elu_fwd(tag, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def elu_bwd(self, tag, y, ygrad, x):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2] if len(y.shape) > 2 else 1
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.elu_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data), list(x.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(rgShape)
        xgrad = xgrad.to(device)
        del rgOut
        del out
        return xgrad

    def sigmoid_fwd(self, tag, x):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2] if len(x.shape) > 2 else 1
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.sigmoid_fwd(tag, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def sigmoid_bwd(self, tag, y, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2] if len(y.shape) > 2 else 1
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.sigmoid_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
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
        nH = x.shape[2] if len(x.shape) > 2 else 1
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
        nH = y.shape[2] if len(y.shape) > 2 else 1
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.layernorm_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(rgShape)
        xgrad = xgrad.to(device)
        del rgOut
        del out
        return xgrad

    def innerproduct_fwd(self, tag, x, bBias, nNumOut, nAxis):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2] if len(x.shape) > 2 else 1
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.innerproduct_fwd(tag, bBias, nNumOut, nAxis, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def innerproduct_bwd(self, tag, y, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2] if len(y.shape) > 2 else 1
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.innerproduct_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(rgShape)
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

    def step(self, iter, enc, dec, trg, e_mask, d_mask):
        rgOut = self.mycaffe.Step(iter, list(enc.detach().cpu().numpy().flatten().data), list(dec.detach().cpu().numpy().flatten().data), list(trg.detach().cpu().numpy().flatten().data), list(e_mask.detach().cpu().numpy().flatten().data), list(d_mask.detach().cpu().numpy().flatten().data))
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

def asNumpyArrayEx(netArray, nOffset, nLen):
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
