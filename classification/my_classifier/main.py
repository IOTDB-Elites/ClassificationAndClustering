import numpy as np

from classification.my_classifier.logistic_regression import LogisticRegression
from classification.preprocess.pre_process import load_train_data, load_train_data_pca
from classification.validation.KFordValidation import k_ford_validation
from sklearn.linear_model import LogisticRegression as SK_LR


if __name__ == '__main__':
    x, y = load_train_data(work_dir='../preprocess/')
    x_pca, y_pca = load_train_data_pca(work_dir='../preprocess/')

    model = LogisticRegression()
    sklearn_model = SK_LR(C=1, penalty='l2', tol=1e-4, solver='lbfgs')


    print(x.shape, y.shape)
    print("sklearn model:")
    k_ford_validation(x, y, sklearn_model)
    print("my model:")
    k_ford_validation(x, y, model)

    print("my model pca:")
    k_ford_validation(x_pca, y_pca, model)
    print("sklearn model pca:")
    k_ford_validation(x_pca, y_pca, sklearn_model)
