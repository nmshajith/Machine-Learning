from collections import Counter
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt( np.sum( (x1-x2)**2 ) )

class KNN():
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predicted_labels = [ self.__predict__(x_test_sample) for x_test_sample in X_test ]
        return np.array(predicted_labels)
    
    def __predict__(self, x_test_sample):
        #Calculate the distance between the x_test_sample and each of the training sample
        distances = [ euclidean_distance(x_test_sample, x) for x in self.X_train ]
        #Find the k nearest sample to the X_test sample
        k_indices = np.argsort(distances)[:self.k]
        #Find the corresponding y label of the nearest k sample
        k_nearest_sample = [self.y_train[kth_index] for kth_index in k_indices]
        #Return the most occuring y label as the index in the k nearest sample
        most_frequent_label_list = Counter(k_nearest_sample).most_common(1)
        return most_frequent_label_list[0][0]