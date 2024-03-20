import anndata
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='data.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    print("Read and Preprocessed Data")

    #X = PCA(heart.X, 100)
    X = PCA(heart.X, 2)
    x = X[:, 0]
    y = X[:, 1]
    print("Performed PCA")


    #Task 2 and 3
    """ silhouette_scores = []  # List to store silhouette scores
    # Iterate over values of k
    for k in range(2, 11):
        km = KMeans(n_clusters=k, init='kmeans++')
        labels = km.fit(X)
        silhouette_avg = km.sil  # Access silhouette score from the KMeans object
        silhouette_scores.append(silhouette_avg)
        print(f"For k = {k}, Silhouette Score: {silhouette_avg}")

    # Plot silhouette scores
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Values of k using KMeans++')
    plt.xticks(range(2, 11))
    plt.grid(True)
    plt.show() """

    #Task 4
    km = KMeans(4, init='random')
    labels = km.fit(X)
    visualize_cluster(x, y, labels)


def visualize_cluster(x, y, clustering):
    for cluster_label in np.unique(clustering):
        # Filter data points belonging to the current cluster
        x_cluster = x[clustering == cluster_label]
        y_cluster = y[clustering == cluster_label]

        # Plot data points for the current cluster
        plt.scatter(x_cluster, y_cluster, label=f'Cluster {cluster_label}', alpha=0.5)

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Cluster Visualization with k = 4 and KMeans++ initilization')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()

if __name__ == '__main__':
    main()
