from __future__ import division
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

breast_cancer = datasets.load_breast_cancer()
X,y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

print('X shape: {}'.format(X.shape))
print('Y Shape: {}'.format(y.shape))


logistic_regression_classifier = LogisticRegression()
logistic_regression_classifier.fit(X_train, y_train)
y_pred = logistic_regression_classifier.predict(X_test)

accuracy = (np.sum(y_test==y_pred))/len(y_test)
print('Accuracy of Logistic Regression: {}'.format(accuracy))


"""
plt.figure(figsize=(8,8))
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, s=20)
plt.xlabel('Radius')
plt.ylabel('Texture')
plt.title('Logistic Regression - Classifier Algorithm (Breast Cancer Dataset)')
plt.show()
"""