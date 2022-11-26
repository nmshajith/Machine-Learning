from __future__ import division
import numpy as np

class KMeansClustering():
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.X_test = None
        self.centroids = None
        self.clusters = None
        self.n_samples = 0
        self.labels = None

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        self.X_test = X_test
        self.n_samples = X_test.shape[0]
        #  [ [] [] []   ] for k in range(self.k)
        self.clusters = [ [] for k in range(self.k)  ] #Each cluster stores the idx of the X_test sample
        #Randomly Initialize k cluster centroid
        random_centroid_idx = np.random.choice(self.n_samples, self.k, replace=False) #Contains each location of each cluster
        self.centroids = [self.X_test[idx] for idx in random_centroid_idx]

        for _ in range(self.max_iter):
            #Assign the data points to cluster centroid
            self._assign_data_points_to_clusters()
            
            old_centroids = self.centroids
            #Recalculate cluster centroid
            self._recalc_cluster_centroid()
            #Is stopping criteria met
            if self._is_converged(old_centroids):
                break
        #Return the labels for each training sample
        #print("Centroids: {}".format(self.centroids))
        #print("Clusters : {}".format(self.clusters))
        self._get_labels_for_each_sample()
        return self.labels

    def _assign_data_points_to_clusters(self):

        for idx, sample in enumerate(self.X_test):
            distance_btw_sample_centroid = [ self._euclidean_distance(centroid, sample) for centroid in self.centroids ]
            cluster_idx = np.argmin(distance_btw_sample_centroid)
            self.clusters[cluster_idx].append(idx)
    
    def _recalc_cluster_centroid(self):
        i = 0
        for idxs in self.clusters:
            self.centroids[i] = np.mean( self.X_test[idxs], axis=0) #Calculate the mean for each feature
    
    def _is_converged(self, old_centroids):
        distance_btw_old_new_centroid = []
        for c in range(self.k):
            distance_btw_old_new_centroid.append( self._euclidean_distance(old_centroids[c], self.centroids[c]) )
        if np.sum(distance_btw_old_new_centroid) == 0:
            return True
        else:
            return False
    
    def _get_labels_for_each_sample(self):
        self.labels = np.zeros(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                #print(sample_idx)
                self.labels[sample_idx] = cluster_idx
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))