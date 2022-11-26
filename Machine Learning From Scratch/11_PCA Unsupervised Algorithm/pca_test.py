from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pca import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

pca = PCA(n_dimensions = 2)
pca.fit(X_train, y_train)
z_test = pca.transform(X_test)
print('X_test shape: {}'.format(X_test.shape))
print('Transform X_test shape (z_test): {}'.format(z_test.shape))


plt.figure(figsize=(8,8))
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.scatter(z_test[:,0], z_test[:,1], c=y_test, cmap=cmap, s=20)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Analysis')
plt.show()