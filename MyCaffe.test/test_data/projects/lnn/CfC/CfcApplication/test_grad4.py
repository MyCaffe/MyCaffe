import torch

def fn_grad(x1, x2):
    return 1/x2, -x1/(x2**2)

def fn(x1, x2):
    y = x1/x2
    return y


# Create a tensor of random values
ta = torch.randn(2, 3, requires_grad=True)
tb = torch.randn(2, 3, requires_grad=True)

# Calculate the SiLU activation of x
y = fn(ta, tb)

# Calculate the gradient of y with respect to x
y.backward(torch.ones_like(ta))

# Print the gradients of x
print(ta.grad)
print(tb.grad)

grad_ta, grad_tb = fn_grad(ta, tb)

# Print the gradients of x
print(grad_ta)
print(grad_tb)

# Check if the two gradients are equal
print(torch.allclose(ta.grad, grad_ta))
print(torch.allclose(tb.grad, grad_tb))
