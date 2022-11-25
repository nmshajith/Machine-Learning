#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import division
import numpy as np

#Prediction 
#(Predict probability for each possible target class-y; The output is the one with highest prob)
#y = argmax [log(P(y)) + log(P(x1|y)) + log(P(x2|y)) + log(P(x3|y)) +...+log(P(xn|y)) ]  #xi -> Stands for each features in a sample 
#P(xi|y) = 1/sqrt(2*pi*(sigma_y**2))*exp(-(xi-u_y)**2 / 2*sigma_y**2 ) 
# #sigma_y -> If y has two 0 and 1; calculate sigma_0 and sigma_1; Calculate each sigma for each feature;
#Say feature = 4; num_of_class=2(Yes and No). For class=Yes, calculate sigma for each feature; For class=Np, calculate sigma for each feature

#P(y) Say y has two class 0&1; P(0) = len(y==0)/len(y); P(1) = len(y==1)/len(y) 


class NaiveBayesClassifier():
    def __init__(self):
        self.mean = None
        self.variance= None
        self.prior = None 
        self.unique_class = None

    def fit(self, X_train, y_train):
        num_of_samples, num_of_features = X_train.shape
        self.unique_class = np.unique(y_train)
        num_of_class = len(self.unique_class)
        #u_y^2-> For each possible target y, there is mean calculated for each feature
        self.mean     = np.zeros((num_of_class, num_of_features),dtype=np.float64) 
        #sigma_y^2 -> For each possible target y, there is variance calculated for each feature
        self.variance = np.zeros((num_of_class, num_of_features),dtype=np.float64) 
        self.prior = np.zeros(num_of_class, dtype=np.float64) #P(y)
        for c in range(num_of_class):
            self.mean[c,:]     = np.mean( X_train[y_train==c], axis=0) #Calculate mean for each column feature
            self.variance[c,:] = np.var(  X_train[y_train==c], axis=0) # Calculate variance for each column feature
            self.prior[c] = len(y_train==c)/len(y_train)
    
    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test] #For each sample in X_test, predict the output
        return y_pred

    def _predict(self,x):
        posteriors = []
        #Loop through the possible y target, and calculate prob for each, and return the target which has the high prob
        for idx,c in enumerate(self.unique_class): 
            prior = np.log(self.prior[idx]) #log(P(y))
            class_conditional = np.sum(np.log(self._pdf(idx,c,x))) #log(P(x1|y))+log(P(x2|y)+...+log(P(xn|y))) ; where n:number of features
            posteriors.append(prior+class_conditional) # Store the prob of each y in the list
        
        return self.unique_class[np.argmax(posteriors)]   #For the computed highest prob, Get the corresponding y label

    def _pdf(self, idx, c, x):
        #Returns the P(x1|y) = Gaussian Distribution; This function returns a vector with length of number_of_features
        #i.e for each feature Gaussian formula is applied with mean, variance of the y, which is being looped
        mean     = self.mean[idx,:] #Gets the corresponding vector of len num_of_samples as the mean
        variance = self.variance[idx,:] #Gets the corresponding vector of len num_of_samples as the variance
        numerator = -((x-mean)**2)
        denominator = 2*variance
        constant = 1/np.sqrt(2*np.pi*variance)
        #print(constant*np.exp(numerator/denominator))
        return constant*np.exp(numerator/denominator) #Returns a vector with length of number_of_features