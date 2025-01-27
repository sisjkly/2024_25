import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin

# Load the Hawk dataset into a DataFrame
df = pd.read_csv('Hawks_good.csv')

# Display the first few rows to understand the structure
print(df.head())

# Assuming 'Feature1' and 'Feature2' are the relevant features for clustering
X = df[['wing', 'weight']].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to plot the clusters and centroids
def plot_clusters(X, centroids, labels, iteration):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for i in range(3):
        points = X[labels == i]
        plt.scatter(points[:, 0], points[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='*', edgecolor='k', label='Centroids')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.grid(True)
    plt.show()

# Randomly initialize centroids by selecting 3 random data points
np.random.seed(14)
initial_indices = np.random.choice(X_scaled.shape[0], 3, replace=False)
centroids = X_scaled[initial_indices]

# Perform K-Means clustering for 4 iterations
for iteration in range(1, 8):
    # Step 1: Assign labels based on closest centroid
    labels = pairwise_distances_argmin(X_scaled, centroids)
    
    # Step 2: Plot the clusters and centroids
    plot_clusters(X_scaled, centroids, labels, iteration)
    
    # Step 3: Update centroids by computing the mean of assigned points
    new_centroids = np.array([X_scaled[labels == i].mean(axis=0) for i in range(3)])
    
    # Check for convergence (if centroids do not change)
    if np.all(centroids == new_centroids):
        print(f'Convergence reached at iteration {iteration}.')
        break
    
    centroids = new_centroids
