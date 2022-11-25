#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import division
import numpy as np 
"""
#f(x) = X.W + b
#Cost function
#J(w,b) = (1/2m) * (i=1tom)Σ (f(x(i)) - y(i))**2
#Gradient Descent
#dj/dw = (1/m)* (i=1tom)Σ (f(x(i)) - y(i))x(i)
#dj/db = (1/m)* (i=1tom)Σ  (f(x(i)) - y(i))
#w = w-alpha*dj/dw
#b = b-alpha*dj/db 
"""
class LinearRegression(): 
    def __init__(self, learning_rate=0.01, num_of_iters=1000):
        self.learning_rate = learning_rate
        self.num_of_iters = num_of_iters
    
    def fit(self, X_train, y_train):
        num_of_samples, num_of_features = X_train.shape
        self.weights = np.zeros(num_of_features) # Shape is (num_of_features, 1) 
        self.bias = 0                            # Shape is scalar
        dj_dw = 0
        for _ in range(self.num_of_iters):
            # Shape of f_x is num_of_samples (Each sample is dot product with weights vector)
            y_pred = np.dot(X_train, self.weights) + self.bias
            #In Below line, np.dot( (75,) and (75,2) ) will output shape of (2,)
            dj_dw = (1/num_of_samples) * np.dot((y_pred-y_train), X_train)
            #In above line, np.dot(X_train, (y_pred-y_train)), won't work, because dot of (75,2) and (75,) won't work
            #print('Shape of dj/dw: {}'.format(dj_dw.shape)) #Shape is (num_of_features,)
            #Below commented line will also give the same output for dj_dw
            """
            for i in range(0, num_of_samples):
                dj_dw += (y_pred[i]-y_train[i])*X_train[i]
            dj_dw = dj_dw/num_of_samples
            """
            dj_db = (1/num_of_samples) * np.sum(y_pred - y_train)
            self.weights = self.weights -(self.learning_rate*dj_dw)
            self.bias    = self.bias    -(self.learning_rate*dj_db)
            #print("W:{}, b:{}".format(self.weights,self.bias))
        
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.weights) + self.bias
        return y_pred
            

