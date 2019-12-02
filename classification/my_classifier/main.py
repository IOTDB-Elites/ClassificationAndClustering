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

    count1 = 0
    x_copy = []
    x_pca_copy = []
    y_copy = []
    y_pca_copy = []
    for k in range(9):
        for i in range(y.shape[0]):
            if y[i] == 1:
                x_copy.append(x[i])
                y_copy.append(1)
                x_pca_copy.append(x_pca[i])
                y_pca_copy.append(1)
    x_copy = np.array(x_copy)
    y_copy = np.array(y_copy)
    x_pca_copy = np.array(x_pca_copy)
    y_pca_copy = np.array(y_pca_copy)

    print(x_copy.shape, y_copy.shape)
    x = np.concatenate((x, x_copy), axis=0)
    y = np.concatenate((y, y_copy), axis=0)
    per = np.random.permutation(x.shape[0])
    x = x[per]
    y = y[per]

    x_pca = np.concatenate((x_pca, x_pca_copy), axis=0)
    y_pca = np.concatenate((y_pca, y_pca_copy), axis=0)
    per = np.random.permutation(x.shape[0])
    x_pca = x_pca[per]
    y_pca = y_pca[per]

    print(x.shape, y.shape)
    print("sklearn model:")
    k_ford_validation(x_pca, y_pca, sklearn_model)
    print("my model:")
    k_ford_validation(x_pca, y_pca, model)

    # print("my model pca:")
    # k_ford_validation(x_pca, y_pca, model)
    # print("sklearn model pca:")
    # k_ford_validation(x_pca, y_pca, sklearn_model)
