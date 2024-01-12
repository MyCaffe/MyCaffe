from contextlib import nullcontext
from curses.ascii import isalnum
from numpy._typing import _128Bit
#from constants import *
import torch
import clr
import os
import numpy as np
import System

path = os.getcwd()
os.chdir('../')
mycaffe_path = os.getcwd() + "\\MyCaffeConnector\\bin\\Debug\\MyCaffeConnector.dll"
clr.AddReference(mycaffe_path)
os.chdir(path)

from MyCaffeConnector import *
from System import Array, Single
import ctypes

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

from System.Runtime.InteropServices import GCHandle, GCHandleType

class MyCaffe():
    def __init__(self):
        pid = os.getpid()
        print("OS PID = {%d}" % (pid))
        self.mycaffe = MyCaffeConnector()
        self.mycaffe.Initialize()
        print("Loaded MyCaffe.")
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.lstm_layers = 2
        self.start = None
        self.end = None
        self.start_b = None

    def get_input_data(self, rgX):
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

    def model_fwd(self, x1, x2, y):
        rgIn = self.get_input_data([x1, x2,y])
        rgOut = self.mycaffe.model_fwd(rgIn)
        outVal = self.get_output_data(rgOut)
        return { "loss" : outVal[0], "predicted_quantiles" : outVal[0] }

    def model_fwd0(self, x1,x2,x3):
        self.start = 0
        self.end = 26
        rgIn = self.get_input_data([x1,x2,x3])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_fwd3(self, x1,x2):
        self.start = 3
        self.end = 26
        rgIn = self.get_input_data([x1,x2])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       
    
    def model_fwd4(self, x1,x2):
        self.start = 4
        self.end = 26
        rgIn = self.get_input_data([x1,x2])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_fwd9(self, x1,x2,x3,x4,x5):
        self.start = 9
        self.end = 26
        rgIn = self.get_input_data([x1,x2,x3,x4,x5])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_fwd12(self, x1,x2,x3,x4):
        self.start = 12
        self.end = 26
        rgIn = self.get_input_data([x1,x2,x3,x4])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_fwd15(self, x1,x2):
        self.start = 15
        self.end = 26
        rgIn = self.get_input_data([x1,x2])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_fwd19(self, x1,x2):
        self.start = 19
        self.end = 26
        rgIn = self.get_input_data([x1,x2])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_fwd23(self, x1,x2):
        self.start = 23
        self.end = 26
        rgIn = self.get_input_data([x1,x2])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_fwd24(self, x1, x2):
        self.start = 24
        self.end = 26
        rgIn = self.get_input_data([x1, x2])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_fwd25(self, x):
        self.start = 25
        self.end = 26
        rgIn = self.get_input_data([x])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_bwd(self, ygrad, x):
        rgIn = self.get_input_data([ygrad])
        rgOut = self.mycaffe.model_bwd(rgIn, self.start, self.end)
        outVal = self.get_output_data(rgOut)
        return outVal       
    
    def model_fwd27(self, x, y):
        self.start_b = 27
        rgIn = self.get_input_data([x,y])
        rgOut = self.mycaffe.model_fwd(rgIn, self.start_b)
        outVal = self.get_output_data(rgOut)
        return outVal       

    def model_bwd27(self, ygrad, xhat):
        rgIn = self.get_input_data([ygrad])
        rgOut = self.mycaffe.model_bwd(rgIn, self.start_b)
        outVal = self.get_output_data(rgOut)
        return outVal       
    
    def model_update(self, nIter):
        self.mycaffe.model_update(nIter)
        
    def model_clear_diffs(self):
        self.mycaffe.model_clear_diffs()

    def model_loss(self):
        return self.mycaffe.CurrentLoss
    
    def model_accuracy(self):
        return self.mycaffe.CurrentAccuracy

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

    def channel_sum_fwd(self, x):
        n = x.shape[0]
        c = x.shape[1] if len(x.shape) > 1 else 1
        h = x.shape[2] if len(x.shape) > 2 else 1
        w = x.shape[3] if len(x.shape) > 3 else 1
        rgIn = x.detach().cpu().numpy().flatten()
        rgOut = self.mycaffe.channel_sum_fwd(n, c, h * w, list(rgIn.data))
        out = asNumpyArray(rgOut)
        outVal = torch.from_numpy(out.reshape((n, c))).to(device)
        return outVal

    def channel_sum_bwd(self, y, h):
        n = y.shape[0]
        c = y.shape[1]
        h1 = h.shape[0]
        rgIn = y.detach().cpu().numpy().flatten()
        rgOut = self.mycaffe.channel_sum_bwd(n, c, h1, list(rgIn.data))
        out = asNumpyArray(rgOut)
        outVal = torch.from_numpy(out.reshape((n, c, h1))).to(device)
        return outVal
        
    def matmul_fwd(self, x1, x2):
        n = x1.shape[0]
        c = x1.shape[1] if len(x1.shape) > 1 else 1
        x1h = x1.shape[2] if len(x1.shape) > 2 else 1
        x1w = x1.shape[3] if len(x1.shape) > 3 else 1
        x2h = x2.shape[2] if len(x2.shape) > 2 else 1
        x2w = x2.shape[3] if len(x2.shape) > 3 else 1
        rgIn1 = x1.detach().cpu().numpy().flatten()
        rgIn2 = x2.detach().cpu().numpy().flatten()
        rgOut = self.mycaffe.matmul_fwd(n, c, x1h, x1w, x2h, x2w, list(rgIn1.data), list(rgIn2.data))
        out = asNumpyArray(rgOut)
        outVal = torch.from_numpy(out.reshape((n, c, x1h, x2w))).to(device)
        return outVal
    
    def matmul_bwd_grad_a(self, y, x1, x2):
        n = y.shape[0]
        c = y.shape[1] if len(y.shape) > 1 else 1
        yh = y.shape[2] if len(y.shape) > 2 else 1
        yw = y.shape[3] if len(y.shape) > 3 else 1
        x1h = x1.shape[2] if len(x1.shape) > 2 else 1
        x1w = x1.shape[3] if len(x1.shape) > 3 else 1
        x2h = x2.shape[2] if len(x2.shape) > 2 else 1
        x2w = x2.shape[3] if len(x2.shape) > 3 else 1
        rgIny = y.detach().cpu().numpy().flatten()
        rgIn1 = x1.detach().cpu().numpy().flatten()
        rgIn2 = x2.detach().cpu().numpy().flatten()
        rgOut = self.mycaffe.matmul_bwd_grad_a(n, c, yh, yw, x1h, x1w, x2h, x2w, list(rgIny.data), list(rgIn1.data), list(rgIn2.data))
        out = asNumpyArray(rgOut)
        outVal = torch.from_numpy(out.reshape((n, c, x1h, x1w))).to(device)
        return outVal
        
    def matmul_bwd_grad_b(self, y, x1, x2):
        n = y.shape[0]
        c = y.shape[1] if len(y.shape) > 1 else 1
        yh = y.shape[2] if len(y.shape) > 2 else 1
        yw = y.shape[3] if len(y.shape) > 3 else 1
        x1h = x1.shape[2] if len(x1.shape) > 2 else 1
        x1w = x1.shape[3] if len(x1.shape) > 3 else 1
        x2h = x2.shape[2] if len(x2.shape) > 2 else 1
        x2w = x2.shape[3] if len(x2.shape) > 3 else 1
        rgOut = self.mycaffe.matmul_bwd_grad_b();
        out = asNumpyArray(rgOut)
        outVal = torch.from_numpy(out.reshape((n, c, x2h, x2w))).to(device)
        return outVal

    def clone_fwd(self, x):
        n = x.shape[0]
        c = x.shape[1] if len(x.shape) > 1 else 1
        xh = x.shape[2] if len(x.shape) > 2 else 1
        xw = x.shape[3] if len(x.shape) > 3 else 1
        rgIn = x.detach().cpu().numpy().flatten()
        rgOut = self.mycaffe.clone_fwd(n, c, xh, xw, list(rgIn.data))
        out = asNumpyArray(rgOut)
        if xw > 1:
            q = torch.from_numpy(out.reshape((n, c, xh, xw))).to(device)
        else:
            q = torch.from_numpy(out.reshape((n, c, xh))).to(device)
        rgOut = self.mycaffe.clone_fwd(n, c, xh, xw, list(rgIn.data))
        out = asNumpyArray(rgOut)
        if xw > 1:
            k = torch.from_numpy(out.reshape((n, c, xh, xw))).to(device)
        else:
            k = torch.from_numpy(out.reshape((n, c, xh))).to(device)
        rgOut = self.mycaffe.clone_fwd(n, c, xh, xw, list(rgIn.data))
        out = asNumpyArray(rgOut)
        if xw > 1:
            v= torch.from_numpy(out.reshape((n, c, xh, xw))).to(device)
        else:
            v = torch.from_numpy(out.reshape((n, c, xh))).to(device)
        return q,k,v
    
    def clone_bwd(self, q,k,v):
        n = q.shape[0]
        c = q.shape[1] if len(q.shape) > 1 else 1
        h = q.shape[2] if len(q.shape) > 2 else 1
        w = q.shape[3] if len(q.shape) > 3 else 1
        rgOut = np.zeros((n, c, h, w), dtype=float).flatten()        
        rgIn = q.detach().cpu().numpy().flatten()
        rgOut = self.mycaffe.clone_bwd(n, c, h, w, list(rgIn.data), list(rgOut.data))
        out = asNumpyArray(rgOut)
        rgIn = k.detach().cpu().numpy().flatten()
        rgOut = self.mycaffe.clone_bwd(n, c, h, w, list(rgIn.data), list(out.data))
        out = asNumpyArray(rgOut)
        rgIn = v.detach().cpu().numpy().flatten()
        rgOut = self.mycaffe.clone_bwd(n, c, h, w, list(rgIn.data), list(out.data))
        out = asNumpyArray(rgOut)
        if w > 1:
            outVal = torch.from_numpy(out.reshape((n, c, h, w))).to(device)
        else:
            outVal = torch.from_numpy(out.reshape((n, c, h))).to(device)
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
        h1 = None
        if (len(hidden) > 0):
            h1 = torch.from_numpy(hidden).float()         
            h1 = h1.reshape((nLayers, nN, nH))
            h1 = h1.to(device)
        c1 = None
        if (len(cell) > 0):
            c1 = torch.from_numpy(cell).float()
            c1 = c1.reshape((nLayers, nN, nH))
            c1 = c1.to(device)

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
        rgHdx = None
        rgCdx = None

        if hgrad != None and cgrad != None:            
            if hgrad[0][0][0].item() != 0 or hgrad[0][0][1].item() != 0 or hgrad[0][0][2].item() != 0 or cgrad[0][0][0].item() != 0 or cgrad[0][0][1].item() != 0 or cgrad[0][0][2].item() != 0:
                rgHdx = list(hgrad.detach().cpu().numpy().flatten().data)
                rgCdx = list(cgrad.detach().cpu().numpy().flatten().data)

        rgOut = self.mycaffe.lstm_bwd(tag, self.lstm_layers, nN, nC, nH, list(y.detach().cpu().numpy().flatten().data), rgH, rgC, list(ygrad.detach().cpu().numpy().flatten().data), rgHdx, rgCdx)
        out = asNumpyArray(rgOut)

        nData = int(out[0])
        nHidden = int(out[1])
        data = out[2:2+nData]
        hidden = out[2+nData:2+nData+nHidden]
        cell = out[2+nData+nHidden:]

        nLayers = self.lstm_layers
        xgrad = torch.from_numpy(data).float()
        xgrad = xgrad.reshape(rgShape)
        xgrad = xgrad.to(device)
        h = None
        if len(hidden) > 0:
            h = torch.from_numpy(hidden).float()
            h = h.reshape((nLayers, nN, nH))
            h = h.to(device)
        c = None
        if len(cell) > 0:
            c = torch.from_numpy(cell).float()
            c = c.reshape((nLayers, nN, nH))
            c = c.to(device)

        del rgOut
        del out
        del data
        del hidden
        del cell
        return [xgrad, h, c]

    def softmax_fwd(self, tag, x, axis):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2] if len(x.shape) > 2 else 1
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.softmax_fwd(tag, axis, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
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

    def tanh_fwd(self, tag, x):
        rgShape = x.shape
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2] if len(x.shape) > 2 else 1
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.tanh_fwd(tag, nN, nC, nH, nW, list(x.detach().cpu().numpy().flatten().data))        
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def tanh_bwd(self, tag, y, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nC = y.shape[1]
        nH = y.shape[2] if len(y.shape) > 2 else 1
        nW = y.shape[3] if len(y.shape) > 3 else 1
        rgOut = self.mycaffe.tanh_bwd(tag, nN, nC, nH, nW, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
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
    
    def innerproduct_diff(self, tag, bBias, diff):
        rgOut = self.mycaffe.innerproduct_diff(tag, bBias)
        out = asNumpyArray(rgOut)
        d = torch.from_numpy(out).float()
        d.reshape(diff.shape)
        del rgOut
        del out
        return d

    def innerproduct_init(self, tag, x, bBias, nNumOut, nAxis, wt, b):
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2] if len(x.shape) > 2 else 1
        nW = x.shape[3] if len(x.shape) > 3 else 1
        nAxis = len(x.shape) - 1
        rgWt = list(wt.detach().cpu().numpy().flatten().data)
        rgB = list(b.detach().cpu().numpy().flatten().data)
        self.mycaffe.innerproduct_init(tag, bBias, nAxis, nNumOut, nN, nC, nH, nW, rgWt, rgB)

    def innerproduct_fwd(self, tag, x, bBias, nNumOut, nAxis):
        if tag == None or tag == "":
            breakpoint()           
        rgShape = x.shape
        if len(rgShape) == 2 and nAxis != 1:
            breakpoint()
        nN = x.shape[0]
        nC = x.shape[1]
        nH = x.shape[2] if len(x.shape) > 2 else 1
        nW = x.shape[3] if len(x.shape) > 3 else 1
        rgIn = list(x.detach().cpu().numpy().flatten().data)
        rgOut = self.mycaffe.innerproduct_fwd(tag, nN, nC, nH, nW, rgIn)        
        out = asNumpyArray(rgOut)
        y = torch.from_numpy(out).float()
        if len(rgShape) > 2:
            if (nNumOut != rgShape[2]):
                rgShape = [rgShape[0], rgShape[1], nNumOut]
        else:
            if (nNumOut != rgShape[1]):
                rgShape = [rgShape[0], nNumOut]
        y = y.reshape(rgShape)
        y = y.to(device)
        del rgOut
        del out
        return y
    
    def innerproduct_bwd(self, tag, y, x, ygrad):
        rgShape = y.shape
        nN = y.shape[0]
        nCy = y.shape[1]
        nCx = x.shape[1]
        nHy = y.shape[2] if len(y.shape) > 2 else 1
        nWy = y.shape[3] if len(y.shape) > 3 else 1
        nHx = x.shape[2] if len(x.shape) > 2 else 1
        nWx = x.shape[3] if len(x.shape) > 3 else 1
        rgOut = self.mycaffe.innerproduct_bwd(tag, nN, nCy, nHy, nWy, nCx, nHx, nWx, list(y.detach().cpu().numpy().flatten().data), list(ygrad.detach().cpu().numpy().flatten().data))
        out = asNumpyArray(rgOut)
        xgrad = torch.from_numpy(out).float()
        xgrad = xgrad.reshape(x.shape)
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
