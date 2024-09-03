import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

# Load the dataset
data = np.load('dataset_problem4.npy')

# Visualize the dataset
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], s=10, c='b', label='Data Points')
plt.title('Visualization of the Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Continue with the rest of the code for clustering

# (a) Using K-Means to cluster the dataset
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data)
kmeans_clusters = kmeans.labels_

# Plot K-Means clustering results
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], c=kmeans_clusters, cmap='viridis', edgecolors='k')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# (b) Spectral Clustering with RBF Kernel
affinity_matrix_rbf = np.exp(-pairwise_distances(data, squared=True))
spectral_rbf = SpectralClustering(n_clusters=3, affinity='precomputed', n_init=100, random_state=0)
spectral_clusters_rbf = spectral_rbf.fit_predict(affinity_matrix_rbf)

# Plot Spectral Clustering results with RBF Kernel
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], c=spectral_clusters_rbf, cmap='viridis', edgecolors='k')
plt.title('Spectral Clustering with RBF Kernel')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# (c) Spectral Clustering with 'nearest neighbors' affinity
k_values = range(1, 21, 2)  # Vary k from 1 to 21, incrementing by 2

for k in k_values:
    affinity_matrix_knn = kneighbors_graph(data, k, mode='connectivity', include_self=True)
    spectral_knn = SpectralClustering(n_clusters=3, affinity='precomputed', n_init=100, random_state=0)
    spectral_clusters_knn = spectral_knn.fit_predict(affinity_matrix_knn)

    # Plot Spectral Clustering results with k-Nearest Neighbors
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=spectral_clusters_knn, cmap='viridis', edgecolors='k')
    plt.title(f'Spectral Clustering with k={k} Nearest Neighbors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
