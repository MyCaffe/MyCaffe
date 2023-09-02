import torch

def softplus_grad(input):
  # Calculate the gradient of SoftPlus wrt the input
  # Return a tensor of the same shape as input containing the gradient values
  output = torch.sigmoid(input)
  return output

def softplus(input):
    # Calculate the SoftPlus
    # Return a tensor of the same shape as input containing the SoftPlus values
    return torch.log(1 + torch.exp(input))


# Create a tensor of random values
x = torch.randn(2, 3, requires_grad=True)

# Calculate the SoftPlus activation of x
y = softplus(x)

# Calculate the gradient of y with respect to x
y.backward(torch.ones_like(x))

# Print the gradients of x
print(x.grad)

# Calculate the direct gradient using softplus_grad()
direct_grad = softplus_grad(x)

# Print the gradients of x
print(direct_grad)

# Check if the two gradients are equal
print(torch.allclose(x.grad, direct_grad))