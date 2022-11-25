#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import division
import numpy as np
from scipy.special import expit
"""
Ridge Classifier
#Prediction
z = W.X+b
f(x) = g(z) = 1/1+exp^-z
#Loss function (for one sample)
L = -y[i]*log(f(x[i])) - (1-y[i]) log(1-f(x[i]))
#Cost Function
J(w,b) = (-1/m) (i=1tom)Σ( y[i]*log(f(x[i])) + (1-y[i])*log(1-f(x[i])) )
#Gradient Descent:
dj/dw = (1/m) (i=1tom)Σ ( (f(x[i])-y[i])*xj[i] ) + (lambda/m)*wj #W-> Array/Vector of num_of_features
dj/db = (1/m) (i=1tom)Σ (f(x[i])-y[i])
"""

class RidgeClassifier():
    def __init__(self, learning_rate=0.001, num_iter=1000, lambda_=4.0):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
        self.threshold = 0.5

    def __sigmoid__(self,y):
        return expit(-y)

    def fit(self, X_train, y_train):
        num_of_samples, num_of_features = X_train.shape
        self.weights = np.zeros(num_of_features)
        self.bias = 0
        for _ in range(self.num_iter):
            y_pred = np.dot(self.weights, X_train.T) + self.bias
            y_pred = self.__sigmoid__(y_pred)
            dj_dw = ((1/num_of_samples) * np.dot((y_pred-y_train), X_train)) + ((self.lambda_/num_of_samples)*self.weights)
            dj_db = (1/num_of_samples) * np.sum(y_pred- y_train)
            self.weights = self.weights-(self.learning_rate*dj_dw)
            self.bias = self.bias-(self.learning_rate*dj_db)
    
    def predict(self, X_test):
        y_pred = np.dot(self.weights, X_test.T)+self.bias
        y_pred = self.__sigmoid__(y_pred)
        y_pred_cls = [1 if y>self.threshold else 0 for y in y_pred]
        return np.array(y_pred_cls)