from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Adaboost import Adaboost

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
y = np.where(y==0, -1, 1)
print("X Shape : {}".format(X.shape))
print("Y Shape: {}".format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

adaboost_classifier = Adaboost(num_of_stumps=5)
adaboost_classifier.fit(X_train, y_train)
y_pred = adaboost_classifier.predict(X_test)

accuracy = np.sum(y_test==y_pred)/len(y_test)
print("Accuracy of Adaboost Classifier Algorithm : {}".format(accuracy))