import numpy as np
from sklearn.mixture import GaussianMixture

# Load the dataset
data = np.load('dataset_problem3.npy')

# Fit a GMM with random initialization
gmm_random_init = GaussianMixture(n_components=5, init_params='random', n_init=1, random_state=0) # Random state set for reproducibility
gmm_random_init.fit(data)

# Get the number of iterations for convergence with random initialization
iterations_random_init = gmm_random_init.n_iter_

# Report the mean values of each cluster for random initialization
means_random_init = gmm_random_init.means_

print("Number of Iterations (Random Initialization):", iterations_random_init)
print("Means of Clusters (Random Initialization):\n", means_random_init)

# Initialize the EM with centers obtained using K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, init='random', n_init=1, max_iter=1)
kmeans.fit(data)
kmeans_centers = kmeans.cluster_centers_

gmm_kmeans_init = GaussianMixture(n_components=5, init_params='kmeans', n_init=1, means_init=kmeans_centers, random_state=0)
gmm_kmeans_init.fit(data)

# Get the number of iterations for convergence with K-means initialization
iterations_kmeans_init = gmm_kmeans_init.n_iter_

# Report the mean values of each cluster for K-means initialization
means_kmeans_init = gmm_kmeans_init.means_

print("Number of Iterations (K-means Initialization):", iterations_kmeans_init)
print("Means of Clusters (K-means Initialization):\n", means_kmeans_init)
