import numpy as np
import torch
from utility import DebugFunction

debug = DebugFunction.apply

def custom_loss(y_true, weights):
        debug = DebugFunction.apply

        DebugFunction.trace(y_true, "y_true")
        y_true = debug(y_true)

        DebugFunction.trace(weights, "weights")
        weights = debug(weights)
                
        captured_returns = weights * y_true              
        DebugFunction.trace(captured_returns, "captured_returns")          
        captured_returns = debug(captured_returns)        

        mean_returns = torch.mean(captured_returns)
        DebugFunction.trace(mean_returns, "mean_returns")
        mean_returns = debug(mean_returns)
        
        mean_returns_sq = torch.square(mean_returns)
        DebugFunction.trace(mean_returns_sq, "mean_returns_sq")
        mean_returns_sq = debug(mean_returns_sq)

        captured_returns_sq = torch.square(captured_returns)
        DebugFunction.trace(captured_returns_sq, "captured_returns_sq")
        captured_returns_sq = debug(captured_returns_sq)

        mean_captured_returns_sq = torch.mean(captured_returns_sq)
        DebugFunction.trace(mean_captured_returns_sq, "mean_captured_returns_sq")
        mean_captured_returns_sq = debug(mean_captured_returns_sq)
        
        mean_captured_returns_sq_minus_mean_returns_sq = mean_captured_returns_sq - mean_returns_sq # + 1e-9
        DebugFunction.trace(mean_captured_returns_sq_minus_mean_returns_sq, "mean_captured_returns_sq_minus_mean_returns_sq")
        mean_captured_returns_sq_minus_mean_returns_sq = debug(mean_captured_returns_sq_minus_mean_returns_sq)
        
        #twofiftytwo = torch.tensor(252.0)

        mean_captured_returns_sq_minus_mean_returns_sqrt = torch.sqrt(mean_captured_returns_sq_minus_mean_returns_sq) 
        DebugFunction.trace(mean_captured_returns_sq_minus_mean_returns_sqrt, "mean_captured_returns_sq_minus_mean_returns_sqrt")
        mean_captured_returns_sq_minus_mean_returns_sqrt = debug(mean_captured_returns_sq_minus_mean_returns_sqrt)

        loss1 = (mean_returns / mean_captured_returns_sq_minus_mean_returns_sqrt) # * torch.sqrt(twofiftytwo))
        DebugFunction.trace(loss1, "loss1")
        loss1 = debug(loss1)
        
        loss = loss1 * -1
        DebugFunction.trace(loss, "loss")
        loss = debug(loss)
        
        return loss

def custom_loss1(y_true, weights):
        debug = DebugFunction.apply

        DebugFunction.trace(y_true, "y_true")
        y_true = debug(y_true)

        DebugFunction.trace(weights, "weights")
        weights = debug(weights)
                
        captured_returns = weights * y_true              
        DebugFunction.trace(captured_returns, "captured_returns")          
        captured_returns = debug(captured_returns)        

        mean_returns = torch.mean(captured_returns)
        DebugFunction.trace(mean_returns, "mean_returns")
        mean_returns = debug(mean_returns)
        
        mean_returns_sq = torch.square(mean_returns)
        DebugFunction.trace(mean_returns_sq, "mean_returns_sq")
        mean_returns_sq = debug(mean_returns_sq)

        captured_returns_sq = torch.square(captured_returns)
        DebugFunction.trace(captured_returns_sq, "captured_returns_sq")
        captured_returns_sq = debug(captured_returns_sq)

        mean_captured_returns_sq = torch.mean(captured_returns_sq)
        DebugFunction.trace(mean_captured_returns_sq, "mean_captured_returns_sq")
        mean_captured_returns_sq = debug(mean_captured_returns_sq)
        
        mean_captured_returns_sq_minus_mean_returns_sq = mean_captured_returns_sq - mean_returns_sq # + 1e-9
        DebugFunction.trace(mean_captured_returns_sq_minus_mean_returns_sq, "mean_captured_returns_sq_minus_mean_returns_sq")
        mean_captured_returns_sq_minus_mean_returns_sq = debug(mean_captured_returns_sq_minus_mean_returns_sq)
        
        twofiftytwo = torch.tensor(252.0)

        mean_captured_returns_sq_minus_mean_returns_sqrt = torch.sqrt(mean_captured_returns_sq_minus_mean_returns_sq) 
        DebugFunction.trace(mean_captured_returns_sq_minus_mean_returns_sqrt, "mean_captured_returns_sq_minus_mean_returns_sqrt")
        mean_captured_returns_sq_minus_mean_returns_sqrt = debug(mean_captured_returns_sq_minus_mean_returns_sqrt)

        loss1 = (mean_returns / mean_captured_returns_sq_minus_mean_returns_sqrt) * torch.sqrt(twofiftytwo)
        DebugFunction.trace(loss1, "loss1")
        loss1 = debug(loss1)
        
        loss = loss1 * -1
        DebugFunction.trace(loss, "loss")
        loss = debug(loss)
        
        return loss

def custom_loss1_grad(y_true, weights, grad_y):
        #debug = DebugFunction.apply

        #DebugFunction.trace(y_true, "y_true")
        #y_true = debug(y_true)

        #DebugFunction.trace(weights, "weights")
        #weights = debug(weights)
                
        captured_returns = weights * y_true              
        #DebugFunction.trace(captured_returns, "captured_returns")          
        #captured_returns = debug(captured_returns)        

        mean_returns = torch.mean(captured_returns)
        #DebugFunction.trace(mean_returns, "mean_returns")
        #mean_returns = debug(mean_returns)
        
        mean_returns_sq = torch.square(mean_returns)
        #DebugFunction.trace(mean_returns_sq, "mean_returns_sq")
        #mean_returns_sq = debug(mean_returns_sq)

        captured_returns_sq = torch.square(captured_returns)
        #DebugFunction.trace(captured_returns_sq, "captured_returns_sq")
        #captured_returns_sq = debug(captured_returns_sq)

        mean_captured_returns_sq = torch.mean(captured_returns_sq)
        #DebugFunction.trace(mean_captured_returns_sq, "mean_captured_returns_sq")
        #mean_captured_returns_sq = debug(mean_captured_returns_sq)
        
        mean_captured_returns_sq_minus_mean_returns_sq = mean_captured_returns_sq - mean_returns_sq # + 1e-9
        #DebugFunction.trace(mean_captured_returns_sq_minus_mean_returns_sq, "mean_captured_returns_sq_minus_mean_returns_sq")
        #mean_captured_returns_sq_minus_mean_returns_sq = debug(mean_captured_returns_sq_minus_mean_returns_sq)
        
        twofiftytwo = torch.tensor(252.0)

        mean_captured_returns_sq_minus_mean_returns_sqrt = torch.sqrt(mean_captured_returns_sq_minus_mean_returns_sq) 
        #DebugFunction.trace(mean_captured_returns_sq_minus_mean_returns_sqrt, "mean_captured_returns_sq_minus_mean_returns_sqrt")
        #mean_captured_returns_sq_minus_mean_returns_sqrt = debug(mean_captured_returns_sq_minus_mean_returns_sqrt)

        loss1 = (mean_returns / mean_captured_returns_sq_minus_mean_returns_sqrt) * torch.sqrt(twofiftytwo)
        #DebugFunction.trace(loss1, "loss1")
        #loss1 = debug(loss1)
        
        loss = loss1 * -1
        #DebugFunction.trace(loss, "loss")
        #loss = debug(loss)

        loss1_grad = grad_y * -1 * torch.sqrt(twofiftytwo)
        mean_captured_returns_sq_minus_mean_returns_sqrt_grad = -1 * mean_returns / mean_captured_returns_sq_minus_mean_returns_sqrt**2 * loss1_grad        
        mean_captured_returns_sq_minus_mean_returns_sq_grad = 0.5 * mean_captured_returns_sq_minus_mean_returns_sq**-0.5 * mean_captured_returns_sq_minus_mean_returns_sqrt_grad
        mean_captured_returns_sq_grad = mean_captured_returns_sq_minus_mean_returns_sq_grad # + 1
        captured_returns_sq_grad = torch.ones_like(captured_returns) / captured_returns.numel() * mean_captured_returns_sq_grad
        mean_returns_sq_grad = -1 * mean_captured_returns_sq_grad
        mean_returns_grad = (2 * mean_returns * mean_returns_sq_grad) + (1 / mean_captured_returns_sq_minus_mean_returns_sqrt * loss1_grad)

        captured_returns_grad_1 = mean_returns_grad / captured_returns.numel()
        captured_returns_grad_2 = 2 * captured_returns * captured_returns_sq_grad
        captured_returns_grad = captured_returns_grad_1 + captured_returns_grad_2
        
        weights_grad_1 = y_true * captured_returns_grad_1
        weights_grad_2 = y_true * captured_returns_grad_2
        weights_grad = weights_grad_1 + weights_grad_2
               
        return weights_grad
        
    
x = torch.tensor([[0.3399, 0.9907, 0.7453, 0.0616, 0.7079],[1.3399, 1.9907, 1.7453, 1.0616, 1.7079]])
yhat = torch.tensor([[0.2, 0.2, 0.2, 0.3, 0.3],[0.5, 0.5, 0.5, 0.5, 0.5]])
yhat.requires_grad = True

loss = custom_loss1(x, yhat)
loss.backward()
print("loss", loss)
print("yhat.grad", yhat.grad)

x_grad = custom_loss1_grad(x, yhat, 1)
print("x_grad", x_grad)

print("done.")