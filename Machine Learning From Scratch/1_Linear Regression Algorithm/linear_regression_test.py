import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

from linear_regression import LinearRegression

X, y = make_regression(n_samples=100, n_features=1, noise=20.0, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)
#print(X.shape)
#print(y.shape) 


linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)

def mean_squared_error(y_test, y_pred):
    return np.mean( (y_test-y_pred)**2)

mean_squared_error = mean_squared_error(y_test, y_pred)
print('Mean Squared Error of Linear Regression: {}'.format(mean_squared_error))

plt.figure(figsize=(6,6))
plt.scatter(X_test, y_test, c='b', marker='o')
plt.plot(X_test, y_pred, c='r')
plt.xlabel('X Samples')
plt.ylabel('Y samples')
plt.title('Linear Regression')
plt.show()
