from __future__ import division
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from Random_forest_classifier import RandomForestClassifier

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
print('X Shape:{}'.format(X.shape))
print('Y shape:{}'.format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=0)

random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train, y_train)
y_pred = random_forest_classifier.predict(X_test)

accuracy = np.sum(y_pred==y_test)/len(y_test)
print('Accuracy of Random Forest Classifier:{}'.format(accuracy))