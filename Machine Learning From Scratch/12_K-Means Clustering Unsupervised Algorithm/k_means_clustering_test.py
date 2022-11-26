from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from k_means_clustering import KMeansClustering
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples = 10000, n_features=2, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

kmeans_clustering = KMeansClustering(k=4)
labels = kmeans_clustering.predict(X_test)

plt.figure(figsize=(8,8))
plt.scatter(X_test[:,0], X_test[:,1], c=labels, s=20)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Unsupervised Algorithm')
plt.show()