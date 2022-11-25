from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from  matplotlib.colors import ListedColormap
from Ridge_regression import Ridge

X, y = datasets.make_regression(n_samples=100, n_features=1,noise=20, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print('X shape: {}'.format(X.shape))
print('Y Shape: {}'.format(y.shape))

"""
plt.figure(figsize=(8,8))
plt.scatter(X[:,0], y, c='b', s=20, marker='o')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Ridge Regression(L2 Penalty)')
plt.show()
"""
ridge_regressor = Ridge()
ridge_regressor.fit(X_train, y_train)
y_pred = ridge_regressor.predict(X_test)

def mse(y_true, y_pred):
    return np.mean( (y_true-y_pred)**2 )

mean_squared_error = mse(y_test, y_pred)
print('Mean Squared Error of Ridge Regression : {}'.format(mean_squared_error))