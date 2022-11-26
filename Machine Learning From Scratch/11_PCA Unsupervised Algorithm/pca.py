from __future__ import division
import numpy as np

class PCA():
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        self.mean = 0
    
    def fit(self, X_train, y_train):
        num_of_samples, num_of_features = X_train.shape
        #Data Preprocessing
        self.mean = np.mean(X_train, axis=0)
        X_train = X_train - self.mean

        #Covariance matrix:
        cov_matrix = np.cov(X_train.T) #np.cov() expects input as shape (number_of_features,num_of_samples)

        #Compute eighen vector:
        eighen_values, eighen_vector = np.linalg.eig(cov_matrix) #eighen_vector are returned as column vector v[:,i]
        #Shape of eighen_vector is (num_of_samples, num_of_samples). Select the first k column to reduce it to k dimensions
        
        #Transform the eighen_vector to row wise
        eighen_vector = eighen_vector.T #Now, each row corresponds to a eighen vector

        #Sort in descending order of eighen values
        idxs = np.argsort(eighen_values)[::-1]
        eighen_value_sort = eighen_values[idxs]
        eighen_vector_sort = eighen_vector[idxs]

        self.components = eighen_vector_sort[:self.n_dimensions]
    
    def transform(self, X_test):
        X_test = X_test- self.mean
        z = np.dot(self.components, X_test.T) 
        #self.components Shape is (2,4)-> 4 is the original dimension, 2 is the reduced dimension; X_test shape: (38,4)
        #z shape is (2,38); Transpose it to make the num_of_samples as the row
        return z.T

