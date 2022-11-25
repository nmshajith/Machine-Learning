from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from naive_bayes_classifier import NaiveBayesClassifier

X, y = datasets.make_classification(n_samples=1000,n_features=10,n_classes=2, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=12)

print("X Shape : {}".format(X.shape))
print("Y Shape : {}".format(y.shape))

"""
plt.figure(figsize=(8,8))
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])
plt.scatter(X[:,0],X[:,1], c=y, cmap=cmap, marker='o')
plt.xlabel('First Feature')
plt.ylabel('Second Feature')
plt.title('Naive Bayes Classifier Algorithm')
plt.legend()
plt.show()
"""
naive_bayes_classifier = NaiveBayesClassifier()
naive_bayes_classifier.fit(X_train, y_train)
y_pred = naive_bayes_classifier.predict(X_test)

accuracy = np.sum(y_test==y_pred)/len(y_test)
print('Accuracy of Naive Bayes Classifier:{}'.format(accuracy))