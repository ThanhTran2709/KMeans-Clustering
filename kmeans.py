import numpy as np


class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()
        self.conv = 0
        self.dist = None
        self.sil = 0
        

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        #clustering = np.zeros(X.shape[0])
        while iteration < self.max_iter:
            clustering = []
            self.dist = self.euclidean_distance(X, self.centroids)
            for i in range(len(X)):
                clustering.append(np.argmin(self.dist[i]))

            old_centroids = self.centroids.copy()
            self.update_centroids(clustering, X)
            new_centroids = self.centroids.copy() 

            converged = True
            for old_centroid, new_centroid in zip(old_centroids, new_centroids):
                if not np.array_equal(old_centroid, new_centroid):
                    converged = True
                    break
            if not converged:
                break
            iteration += 1
            
        self.sil = self.silhouette(clustering, X)
        self.conv = iteration
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        temp = {}
        for cluster in np.unique(clustering):
            temp[cluster] = []
        for data, cluster in zip(X,clustering):
            temp[cluster].append(data)
        for cluster in temp: 
            self.centroids[cluster] = np.average(temp[cluster], axis = 0)

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            data_points = np.random.choice(X.shape[0], self.n_clusters, replace= False)
            self.centroids = X[data_points, :].copy()

        elif self.init == 'kmeans++':
            data_points = np.random.choice(X.shape[0], 1, replace = False)
            self.centroids = X[data_points, :].copy()

            for k in range(1, self.n_clusters):
                distances = self.euclidean_distance(self.centroids, X)
                weights = np.min(distances, axis=0)
                weights /= np.sum(weights)

                next_centroid_index = np.random.choice(X.shape[0], 1, replace = False, p = weights)
                next_centroid = X[next_centroid_index,:].copy()
                self.centroids = np.append(self.centroids,next_centroid,axis = 0) 
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # Expand dimensions to enable broadcasting
        X1_expand = np.expand_dims(X1, axis=1)
        X2_expand = np.expand_dims(X2, axis=0)

        # Compute squared differences element-wise
        squared_diff = (X1_expand - X2_expand) ** 2

        # Sum along the feature axis and take the square root
        dist = np.sqrt(np.sum(squared_diff, axis=2))
        
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        sil = 0
        for i in range(len(X)):
            # Calculate average distance to other points in the same cluster (a)
            cluster_points = X[clustering == clustering[i]]
            a = np.mean(np.linalg.norm(cluster_points - X[i], axis=1))

            # Calculate average distance to points in the nearest neighboring cluster (b)
            other_cluster_points = X[clustering != clustering[i]]
            b = np.min(np.mean(np.linalg.norm(other_cluster_points - X[i], axis=1)))

            s = (b-a)/max(a,b)
            sil = sil + s
            
        return sil/(len(X))