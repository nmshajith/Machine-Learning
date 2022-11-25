from __future__ import division
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from Ridge_regression_classifier import RidgeClassifier

breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
print('X Shape:{}'.format(X.shape))
print('Y shape:{}'.format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size =0.75, test_size=0.25, random_state=0)

ridge_classifier = RidgeClassifier()
ridge_classifier.fit(X_train, y_train)
y_pred = ridge_classifier.predict(X_test)

accuracy = np.sum(y_pred==y_test)/len(y_test)
print('Accuracy of Ridge Classifier: {}'.format(accuracy))