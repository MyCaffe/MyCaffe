import torch

def lecun_grad(input):
    # Compute the gradient of the LeCun function with respect to input
    # Use the chain rule and the derivative of tanh
    # Return a tensor of the same shape as input containing the gradient values
    return 1.7159 * 2/3 * (1 - torch.tanh(2/3 * input) ** 2)

def lecun(input):
    # Calculate the LeCun
    # Return a tensor of the same shape as input containing the LeCun values
    return 1.7159 * torch.tanh(2/3 * input)


# Create a tensor of random values
x = torch.randn(2, 3, requires_grad=True)

# Calculate the LeCun activation of x
y = lecun(x)

# Calculate the gradient of y with respect to x
y.backward(torch.ones_like(x))

# Print the gradients of x
print(x.grad)

# Calculate the direct gradient using lecun_grad()
direct_grad = lecun_grad(x)

# Print the gradients of x
print(direct_grad)

# Check if the two gradients are equal
print(torch.allclose(x.grad, direct_grad))