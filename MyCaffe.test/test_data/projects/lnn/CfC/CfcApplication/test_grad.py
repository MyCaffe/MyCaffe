import torch

def fn_grad(ta, ts, tb):
    return ts, ta, torch.ones(tb.shape)

def fn(ta, ts, tb):
    return tb + ta * ts


# Create a tensor of random values
ta = torch.randn(2, 3, requires_grad=True)
ts = torch.randn(2, 3, requires_grad=True)
tb = torch.randn(2, 3, requires_grad=True)

# Calculate the SiLU activation of x
y = fn(ta, ts, tb)

# Calculate the gradient of y with respect to x
y.backward(torch.ones_like(ta))

# Print the gradients of x
print(ta.grad)
print(ts.grad)
print(tb.grad)

grad_ta, grad_ts, grad_tb = fn_grad(ta, ts, tb)

# Print the gradients of x
print(grad_ta)
print(grad_ts)
print(grad_tb)

# Check if the two gradients are equal
print(torch.allclose(ta.grad, grad_ta))
print(torch.allclose(ts.grad, grad_ts))
print(torch.allclose(tb.grad, grad_tb))
