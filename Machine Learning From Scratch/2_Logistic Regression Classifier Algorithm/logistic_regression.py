#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import division
import numpy as np
#import math
from scipy.special import expit
#from doubledouble import DoubleDouble
#Prediction:
#z = W.X+b
#f(x) = g(z) = 1/ (1+e^-z)

#Loss Function (for a single sample)
#L = -y(i) log( f(x(i)) ) - (1-y(i) * log(1-f(x(i)) ) )

#Cost Function
#J(w,b) = -(1/m) (i=1tom)Σ [ y(i) log( f(x(i)) ) + (-y(i) * log(1-f(x(i)) ) ]

#Gradient Descent (For each weight corresponding to a feature)
#wj = wj - alpha*dj/dwj
#b =  b  - alpha*dj/db

#dj/dwj = 1/m (i=1tom)Σ (f(x(i)) - y(i))*xj(i) #Array of n samples
#dj/db  = 1/m (i=1tom)Σ (f(x(i)) - y(i))


class LogisticRegression():
    def __init__(self, learning_rate=0.0001, num_of_iter=1000):
        self.learning_rate = learning_rate
        self.num_of_iter = num_of_iter
        self.weights = None
        self.bias = None
        self.threshold = 0.5

    def __sigmoid__(self, linear_y):
        #print(type(linear_y))
        #linear_y = linear_y.astype(np.longdouble)
        return expit(-linear_y)

    def fit(self, X_train, y_train):
        num_of_samples, num_of_features = X_train.shape
        self.weights = np.zeros(num_of_features)
        self.bias = 0

        for _ in range(self.num_of_iter):
            linear_model = np.dot(self.weights, X_train.T) + self.bias
            
            y_pred = self.__sigmoid__(linear_model)
            dj_dw = (1/num_of_samples) * np.dot( (y_pred-y_train), X_train)
            dj_db = (1/num_of_samples) * np.sum(y_pred-y_train)
            self.weights = self.weights -(self.learning_rate*dj_dw)
            self.bias    = self.bias    -(self.learning_rate*dj_db)

    def predict(self, X_test):
        linear_y_pred = np.dot(self.weights, X_test.T)+self.bias
        y_pred = self.__sigmoid__(linear_y_pred)
        y_pred = [1 if y>self.threshold else 0 for y in y_pred]
        return np.array(y_pred)