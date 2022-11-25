from __future__ import division
from collections import Counter
import numpy as np

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        if self.value == None:
            return False
        else:
            return True

class DecisionTreeClassifier():
    def __init__(self, max_depth=100, min_split_samples=2, n_feature=None):
        self.max_depth = max_depth
        self.min_split_samples = min_split_samples
        self.n_feature = n_feature #Number of features to be looked at, when splitting the tree/node
        self.root = None #Root of the tree
    
    def fit(self, X_train, y_train):
        self.root = self._build_tree(X_train, y_train, depth=0)

    def _most_common_label(self, y):
        most_common_label = Counter(y).most_common(1)[0][0]
        return most_common_label

    def get_split_data(self, X_train_per_feature, threshold):
        left_indices = np.argwhere(X_train_per_feature<=threshold).flatten()
        right_indices = np.argwhere(X_train_per_feature>threshold).flatten()
        return (left_indices, right_indices)
    
    def compute_entropy(self,y):
        """
            Entropy E = -(x=1tonum_of_unique_labels)Summation p(x)*log2p(x)
        """
        hist = np.bincount(y)
        prob_vector = hist/len(y)
        #return -np.sum( prob_vector*np.log2(prob_vector) ) 
        return -np.sum (prob*np.log2(prob) for prob in prob_vector if prob>0)

    def information_gain(self, X_train_per_feature, y_train, threshold):
        """
            For feature column in X_train, and its threshold values, get the left indices, and right indices
            Information gain = H(p1_root) - (W_left*H(p1_left) + W_right*H(p1_right))
            H -> Entropy (Call compute_entropy function)
            W_left = number of elements in left node / number of elements in root node
            W_right = number of elements in right node/ number of elements in root node
        """
        left_indices, right_indices = self.get_split_data(X_train_per_feature, threshold)
        if len(left_indices)==0 or len(right_indices)==0:
            return 0
        root_entropy     = self.compute_entropy(y_train)
        left_entropy     = self.compute_entropy(y_train[left_indices])
        right_entropy    = self.compute_entropy(y_train[right_indices])
        W_left           = len(left_indices)/len(y_train)
        W_right          = len(right_indices)/len(y_train)
        information_gain = root_entropy - ((W_left*left_entropy) + (W_right*right_entropy))
        return information_gain

    def get_best_feature_and_threshold(self, X_train, y_train):
        """
            Loop through each feature and its unique values from its feature_column
            For a set of (feature, threshold), compute the information gain, for that column
            Store the combination (feature, threshold) which has the best information gain
        """
        best_information_gain = -1
        for feature_idx in range(len(X_train.T)):
            threshold_list = np.unique(X_train.T[feature_idx])
            for threshold in threshold_list:
                information_gain = self.information_gain( X_train.T[feature_idx], y_train, threshold)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_tree(self, X_train,y_train, depth=0):
        """
            Recursively build the tree, and find the best feature, threshold for that X_train, y_train
            If stopping criteria is met, then return the Node Datastructure with the prediction as the most common label in that leaf node
        """
        num_of_samples, num_of_features = X_train.shape
        n_unique_labels = len(np.unique(y_train))
        #Stopping criteria:
        #Return the Leaf node, with its value as the most common target y
        if (depth >=self.max_depth or
            num_of_samples <= self.min_split_samples or
            n_unique_labels == 1):
            leaf_value = self._most_common_label(y_train)
            print("Leaf Value:{}, depth:{}".format(leaf_value, depth))
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self.get_best_feature_and_threshold(X_train, y_train)
        #print("Depth: {}; Best Feature: {}; Threshold:{}".format(depth, best_feature, best_threshold))
        left_indices, right_indices = self.get_split_data(X_train.T[best_feature], best_threshold)
        #print('Going to build left node')
        left_node  = self._build_tree(X_train[left_indices,:], y_train[left_indices], depth+1)
        #print('Finished building left node at depth:{}'.format(depth+1))
        #print('Going to build right node')
        right_node = self._build_tree(X_train[right_indices,:], y_train[right_indices], depth+1)
        #print('Finished building right node at depth:{}'.format(depth+1))
        return Node(best_feature, best_threshold, left_node, right_node)
    
    def predict(self, X_test):
        """
            For each sample vector, traverse through the tree, and find the leaf node value
            Ex: If Vector is (is_rainy:10.5, is_sunny:1, is_cloudy:0.5) etc. Get back the best feature stored of root, and traverse through
            each node with that node's best feature and threshold, to go to the leaf node
        """
        y_pred = [self._traverse_tree(x, self.root) for x in X_test]
        return np.array(y_pred)
    
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): #If the leaf node is reached, return the stored value
            return node.value
        if x[node.feature] > node.threshold: #For that node's best feature, if the value in the vector is below threshold, go to left node, else right node
            return self._traverse_tree(x, node.right)
        else:
            return self._traverse_tree(x, node.left)
