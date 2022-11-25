from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
#Import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#Import KNN File and its Class
from KNN import KNN


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

"""
plt.figure(figsize=(8,8))
cmap= ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, s=20)
plt.xlabel('Sepal Width in cm')
plt.ylabel('Petal length in cm')
plt.title('Iris Dataset')
plt.show()
"""

knn_classifier = KNN(k=3)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = (np.sum(y_test == y_pred))/len(y_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion_Matrix:\n {}'.format(cm))
print('Accuracy: {}'.format(accuracy))