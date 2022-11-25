#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import division
import numpy as np
"""
#Prediction:
y = W.X+b
m - No.of samples
n - No.of features
#Cost function (Minimise it)
J(w,b) = ( (i=1tom)Σ ( f(x[i])-y[i] )**2 ) + ((lambda/2m)*(j=1ton)Σwj**2)
i.e Adding a high penalty when any weight is high

#Gradient Descent:
wj = wj - alpha*dj/dwj -> W: Array/Vector of number of features
b  = b  - alpha*dj/db
dj/dwj = (1/m) * (i=1tom)Σ( (f(x[i])-y[i])*xj[i] )+ (lambda/m)*wj
dj/db  = (1/m) * (i=1tom)Σ  (f(x[i])-y[i])
"""
class Ridge():
    def __init__(self, learning_rate=0.01, num_of_iters=1000, lambda_ = 10.0):
        self.learning_rate = learning_rate
        self.num_of_iters = num_of_iters
        self.weights = None
        self.bias = None
        self.lambda_ = lambda_
    
    def fit(self, X_train, y_train):
        num_of_samples, num_of_features = X_train.shape
        self.weights = np.zeros(num_of_features)
        self.bias = 0
        for _ in range(self.num_of_iters):
            y_pred = np.dot(self.weights, X_train.T)+self.bias
            dj_dw = ((1/num_of_samples) *np.dot((y_pred-y_train), X_train)) + ((self.lambda_/num_of_samples)*self.weights)
            dj_db = (1/num_of_samples) * np.sum(y_pred-y_train)
            self.weights = self.weights - self.learning_rate*dj_dw
            self.bias = self.bias * self.learning_rate*dj_db
    
    def predict(self, X_test):
        y_pred = np.dot(self.weights, X_test.T) + self.bias
        return y_pred