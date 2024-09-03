import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = np.load('dataset_problem3.npy')

# Define pairs of coordinates for visualization
coordinate_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]

# Create subplots for visualization
plt.figure(figsize=(12, 8))

for i, (x, y) in enumerate(coordinate_pairs, start=1):
    plt.subplot(2, 2, i)
    plt.scatter(data[:, x], data[:, y])
    plt.title(f'Coordinates ({x}, {y})')

plt.show()
