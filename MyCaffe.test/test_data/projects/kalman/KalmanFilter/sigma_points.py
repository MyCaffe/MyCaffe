import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import MerweScaledSigmaPoints

# Create sigma points object
sigma_points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2.0, kappa=0)

# Example state mean and covariance
x = np.array([2, 2])  # mean state
P = np.array([[1, 0.5], [0.5, 1]])  # covariance matrix

# Generate sigma points
points = sigma_points.sigma_points(x, P)

# Plot the points
plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], color='dodgerblue', s=100, edgecolors='black', label='Sigma Points')  # nicer color and outlined
plt.scatter(x[0], x[1], color='red', s=200, edgecolors='black', label='Mean')  # highlight mean

# Annotate points
for i, point in enumerate(points):
    plt.annotate(f'Point {i}', (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Visualization of Merwe Scaled Sigma Points')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

print('done!')