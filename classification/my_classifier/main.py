import numpy as np
from classification.my_classifier.logistic_regression import LogisticRegression
from sklearn import datasets
from classification.validation.KFordValidation import k_ford_validation
from sklearn.linear_model import LogisticRegression as SK_LR

iris = datasets.load_iris()
X = iris.data[0:100]
Y = iris.target[0:100]

if __name__ == '__main__':
    x = np.array([[1, 2], [1.1, 1.9], [1.02, 2.05], [2, 1], [1.9, 1.1], [2.3, 0.9]])
    a = np.array([[1], [1]])
    b = np.array([[1], [1]])
    y = np.array([[0], [0], [0], [1], [1], [1]])

    model = LogisticRegression()
    sklearn_model = SK_LR(C=1, penalty='l2', tol=1e-4, solver='lbfgs')

    print("my model:")
    k_ford_validation(X, Y, model)
    print("sklearn model:")
    k_ford_validation(X, Y, sklearn_model)
