import os
cwd = os.getcwd()
import torch
from torch import nn
import numpy as np
from utility import DebugFunction
from tft_torch.tft import LayerNormEx

os.chdir(cwd)
print(os.getcwd())

debug = DebugFunction.apply

def layernorm_forward(x, eps=1e-10):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    
    N, D = x.shape
    xT = x.T
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    feature_mean = np.mean(xT, axis = 0) # feature mean (D,)
    feature_var = np.var(xT, axis = 0) # feature variance (D,)
    
    scaled_xT = xT - feature_mean
    normalize_xT = (xT - feature_mean)/np.sqrt(feature_var + eps) 
    
    normalize_x = normalize_xT.T
    scaled_x = scaled_xT.T
    
    out = normalize_x        

    cache = {
                'scaled_x' : scaled_x,  # (N, D)
                'ivar' : 1./np.sqrt(feature_var + eps), # (D,)
                'sqrtvar' : np.sqrt(feature_var + eps)  # (D,)
            }   
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache

def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx = None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    N, D = dout.shape
    ivar = cache.get('ivar')
    scaled_x = cache.get('scaled_x')
    sqrtvar = cache.get('sqrtvar')    

    doutT = dout.T
    scaled_xT = scaled_x.T

    sumDoutT = np.sum(doutT, axis=0)

    dx.T = (1 / N) * (1/sqrtvar * ((N * doutT)) - np.sum(doutT, axis=0) - ((scaled_xT) * np.square(ivar) * np.sum(doutT*scaled_x, axis=0))) 
    
    dx = dx.T
    #print(dx.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

strPathWt = "data/favorita/weights/"
strPath = "test/iter_0/"
DebugFunction.set_output_path(0)

x = np.load(strPath + "test4_gan.layernorm.x.npy")

y, cache = layernorm_forward(x)

dy = np.load(strPath + "test4_gan.layernorm.y.grad.npy")

dx = layernorm_backward(dy, cache)


layernorm = nn.LayerNorm(x.shape[1], eps=1e-10)
y1 = layernorm(x)

DebugFunction.trace(y1, "y1.tt")
y1 = debug(y1)

p0 = y1.clone()
p0 = p0 * 0 + 1

loss = (y1 - p0).sum()

loss.backward()




















#x = np.array([[1.,3.,8.],[10.,11.,19.]])
#x = torch.from_numpy(x).type(torch.DoubleTensor);
#x.requires_grad = True


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

x = np.load(strPath + "test4_gan.layernorm.x.npy")
x = torch.from_numpy(x).to(device)
x.requires_grad = True;

N,D = x.shape

DebugFunction.trace(x, "ln_x")
x = debug(x)

#step1
mu = 1./D * torch.sum(x, axis = 1)
mu = mu.unsqueeze(1)
DebugFunction.trace(mu, "ln_mu");
#step2
xmu = x - mu
DebugFunction.trace(xmu, "ln_xmu");
#step3
sq = xmu ** 2
DebugFunction.trace(sq, "ln_sq");
#step4
var = 1./D * torch.sum(sq, axis=1)
var = var.unsqueeze(1)
DebugFunction.trace(var, "ln_var");
#step5
var += 1e-05
sqrtvar = torch.sqrt(var)
DebugFunction.trace(sqrtvar, "ln_sqrtvar");
#step6
ivar = 1./sqrtvar
DebugFunction.trace(ivar, "ln_ivar");
#step7
xhat = xmu * ivar
DebugFunction.trace(xhat, "ln_xhat");
print(xhat)

# ========GRAD========

dxhat = np.load(strPath + "test4_gan.layernorm.y.grad.npy")
dxhat = torch.from_numpy(dxhat).to(device)
dxhat.requires_grad = True;
DebugFunction.trace(dxhat, "ln_dxhat");

#step7
dxhat_xmu = dxhat * xmu
DebugFunction.trace(dxhat_xmu, "ln_dxhat_xmu");

divar = torch.sum(dxhat_xmu, axis=1)
divar = divar.unsqueeze(1)
DebugFunction.trace(divar, "ln_divar");

dxmu1 = dxhat * ivar
DebugFunction.trace(dxmu1, "ln_dxmu1");
#step6
dsqrtvar = -1. / (sqrtvar**2) * divar
DebugFunction.trace(dsqrtvar, "ln_dsqrtvar");
#step5
dvar = 0.5 * 1. / torch.sqrt(var) * dsqrtvar
DebugFunction.trace(dvar, "ln_dvar");
#step4
ones = torch.ones((N,D)).to(device)
dsq = 1. / D * ones * dvar
DebugFunction.trace(dsq, "ln_dsq");

#step3
dxmu2 = 2 * xmu * dsq
DebugFunction.trace(dxmu2, "ln_dxmu2");

#step2
dx1 = (dxmu1 + dxmu2)
DebugFunction.trace(dx1, "ln_dx1");

dmu = -1 * torch.sum(dxmu1 + dxmu2, axis=1)
dmu = dmu.unsqueeze(1)
DebugFunction.trace(dmu, "ln_dmu");

#step1
dx2 = 1. / D * ones * dmu
DebugFunction.trace(dx2, "ln_dx2");

#step0
dx = dx1 + dx2
DebugFunction.trace(dx, "ln_dx");

sumdx = dx.sum()

print(dx)


layernorm = LayerNormEx([D])
x1 = x.clone()

DebugFunction.trace(x1, "x1")
x1 = debug(x1)

xhat2 = layernorm(x1)

DebugFunction.trace(xhat2, "xhat2")
xhat2 = debug(xhat2)

p0 = xhat2.clone()
p0 = p0 * 0 + 1

loss = (xhat2 - p0).sum() * 1000000000

loss.backward()

x1_grad = np.load(strPath + "x1.grad.npy")
x1_grad = torch.from_numpy(x1_grad).to(device)

x1_grad_sum = x1_grad.sum()
diff = sumdx - x1_grad_sum

xhat2_grad = np.load(strPath + "xhat2.grad.npy")
xhat2_grad = torch.from_numpy(xhat2_grad).to(device)
diff2 = (dxhat - xhat2_grad).sum()

print(diff)


