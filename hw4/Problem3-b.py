import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = np.load('dataset_problem3.npy')

inertias = []

for i in range(10):
    if i == 0:
        kmeans = KMeans(n_clusters=5, init='random', random_state=0, max_iter=1, n_init=1)
    else:
        kmeans = KMeans(n_clusters=5, init=kmeans.cluster_centers_, max_iter=1, n_init=1)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Plot within-cluster scatter versus iterations
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of Iterations')
plt.ylabel('Within-Cluster Scatter')
plt.title('K-means Convergence')
plt.show()