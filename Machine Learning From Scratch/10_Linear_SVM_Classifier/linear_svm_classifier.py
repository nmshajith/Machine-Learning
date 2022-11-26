from __future__ import division
import numpy as np


class LinearSVMClassifier():
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_param
        self.num_iter = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        y_train = np.where(y_train<=0, -1, 1)
        num_of_samples, num_of_features = X_train.shape
        self.weights = np.zeros(num_of_features)
        self.bias = 0

        for _ in range(self.num_iter):
            for idx, xi in enumerate(X_train):
                f_xi = np.dot( xi, self.weights) - self.bias
                condition = y_train[idx]*f_xi
                if condition>=1:
                    dji_dw = 2*self.lambda_*self.weights
                    dji_db = 0
                else:
                    dji_dw = (2*self.lambda_*self.weights) - (np.dot(y_train[idx], xi))
                    dji_db = y_train[idx]
                self.weights = self.weights - (self.learning_rate*dji_dw)
                self.bias = self.bias - (self.learning_rate*dji_db)
    
    def predict(self, X_test):
        y_pred = np.dot(self.weights, X_test.T) - self.bias
        return np.sign(y_pred)
