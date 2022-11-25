from __future__ import division
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decision_tree_classifier import DecisionTreeClassifier
import numpy as np

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)
#print(X_train.T[29])
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
#print(y_pred)
accuracy = np.sum(y_pred == y_test)/len(y_test)
print('Accuracy of Decision Tree Classifier :{}'.format(accuracy))