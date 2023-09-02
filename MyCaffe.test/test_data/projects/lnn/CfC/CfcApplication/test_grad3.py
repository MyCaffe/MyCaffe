import torch

def fn_grad(ff1, ti, ff2):
    return (1.0 - ti), ff2 - ff1, ti

def fn(ff1, ti, ff2):
    return ff1 * (1.0 - ti) + ti * ff2


# Create a tensor of random values
ff1 = torch.randn(2, 3, requires_grad=True)
ti = torch.randn(2, 3, requires_grad=True)
ff2 = torch.randn(2, 3, requires_grad=True)

# Calculate the SiLU activation of x
y = fn(ff1, ti, ff2)

# Calculate the gradient of y with respect to x
y.backward(torch.ones_like(ti))

# Print the gradients of x
print(ff1.grad)
print(ti.grad)
print(ff2.grad)

grad_ff1, grad_ti, grad_ff2 = fn_grad(ff1, ti, ff2)

# Print the gradients of x
print(grad_ff1)
print(grad_ti)
print(grad_ff2)

# Check if the two gradients are equal
print(torch.allclose(ff1.grad, grad_ff1))
print(torch.allclose(ti.grad, grad_ti))
print(torch.allclose(ff2.grad, grad_ff2))
