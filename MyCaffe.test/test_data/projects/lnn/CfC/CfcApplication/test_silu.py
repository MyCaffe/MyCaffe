import torch

def silu_grad(input):
  # Calculate the gradient of torch.sigmoid(input) with respect to input
  # Use the chain rule: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
  # Return a tensor of the same shape as input containing the gradient values
  output = torch.sigmoid(input)
  return output * (1 + input * (1 - output))

def silu_grad_expanded(input):
  # Calculate the gradient of torch.sigmoid(input) with respect to input
  # Use the chain rule: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
  # Return a tensor of the same shape as input containing the gradient values
  silu1 = silu(input)
  sigmoid1 = torch.sigmoid(input)
  return sigmoid1 + silu1 - (input * sigmoid1 * sigmoid1)

def silu(input):
    # Calculate the sigmoid weighted input linear unit (SiLU)
    # Return a tensor of the same shape as input containing the SiLU values
    return input * torch.sigmoid(input)


# Create a tensor of random values
x = torch.randn(2, 3, requires_grad=True)

# Calculate the SiLU activation of x
y = silu(x)

# Calculate the gradient of y with respect to x
y.backward(torch.ones_like(x))

# Print the gradients of x
print(x.grad)

# Calculate the direct gradient using silu_grad()
direct_grad = silu_grad(x)

# Print the gradients of x
print(direct_grad)

direct_grad2 = silu_grad_expanded(x)
print(direct_grad2)

# Check if the two gradients are equal
print(torch.allclose(x.grad, direct_grad))