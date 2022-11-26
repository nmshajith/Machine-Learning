from __future__ import division
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn import datasets
from linear_svm_classifier import LinearSVMClassifier

X,y = datasets.make_classification(n_samples=100, n_features=20, random_state=0)
print("X Shape : {}".format(X.shape))
print("Y Shape : {}".format(y.shape))
y = np.where(y==0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)


linear_svm_classifier = LinearSVMClassifier(learning_rate=0.001, lambda_param=3.0, n_iters=1000)
linear_svm_classifier.fit(X_train, y_train)

y_pred = linear_svm_classifier.predict(X_test)
#print(y_test)
#print(y_pred)
acccuracy = np.sum(y_pred==y_test)/len(y_test)
print('Accuracy of Linear SVM Classifier: {}'.format(acccuracy))