from __future__ import division
from collections import Counter
import numpy as np

import sys
import os
 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
dt_folder = os.path.join(parent, '8_Decision Tree Classifier Algorithm')

sys.path.append(dt_folder)
from decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier():
    def __init__(self, n_trees=100, min_num_samples=2, n_features=10, max_depth=100):
        self.n_trees = n_trees
        self.min_num_samples = min_num_samples
        self.n_features = n_features
        self.max_depth = max_depth
        self.trees = [] #List to store the objects of all the Decision Tree Class

    def get_subset_data(self, X, y):
        n_samples, n_features = X.shape
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return (X[idxs], y[idxs])

    def fit(self, X_train, y_train):
        self.trees = []

        for _ in range(self.n_trees):
            dt_tree = DecisionTreeClassifier( max_depth=100, min_split_samples=2)
            X_subset, y_subset = self.get_subset_data(X_train, y_train)
            dt_tree.fit(X_subset, y_subset)
            self.trees.append(dt_tree)
    
    def predict(self, X_test):
        y_pred_list = [dt_tree.predict(X_test) for dt_tree in self.trees]
        #y_pred = [ [1111] [0000] [0101]] -> Each internal list corresponds to y_pred from each Decision Tree
        y_pred_list = np.swapaxes(y_pred_list, 0, 1)
        #y_pred = [ [100] [101] [100] [101]  ] -> Each internal list corresponds to labels for each X_test vector
        #Below: Get the majority vote for each vector
        y_pred = [Counter(y_pred).most_common(1)[0][0]  for y_pred in y_pred_list]
        return np.array(y_pred)
    
    def _predict_each_vector_op(self, x):
        predicted_label = [] #List to store the op target value, from each Decision Tree
        for tree_obj in self.trees:
            tree_obj.predict()
