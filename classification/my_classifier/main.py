import numpy as np

from classification.my_classifier.logistic_regression import LogisticRegression
from classification.preprocess.pre_process import load_train_data, load_train_data_pca
from classification.validation.KFordValidation import k_ford_validation
from sklearn.linear_model import LogisticRegression as SK_LR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    x, y = load_train_data(work_dir='../preprocess/')
    x_pca, y_pca = load_train_data_pca(work_dir='../preprocess/')

    model = LogisticRegression()
    sklearn_model = SK_LR(C=1, penalty='l2', tol=1e-4, solver='lbfgs')
    svm_model = SVC(tol=1e-4)
    decision_tree_model = DecisionTreeClassifier(min_samples_split=2)

    layer = []
    for i in range(6):
        layer.append(100)
    layer_tuple = tuple(layer)
    mlp_model = MLPClassifier(hidden_layer_sizes=layer_tuple, max_iter=10000, tol=1e-8)

    print(x.shape, y.shape)
    print("sklearn model:")
    k_ford_validation(x, y, sklearn_model)
    print("svm model:")
    k_ford_validation(x, y, svm_model)
    print("decision tree model:")
    k_ford_validation(x, y, decision_tree_model)
    print("mlp model:")
    k_ford_validation(x, y, mlp_model)
    print("my model:")
    k_ford_validation(x, y, model)

    print("sklearn model pca:")
    k_ford_validation(x_pca, y_pca, sklearn_model)
    print("svm model pca:")
    k_ford_validation(x_pca, y_pca, svm_model)
    print("decision tree model pca:")
    k_ford_validation(x_pca, y_pca, decision_tree_model)
    print("mlp model pca:")
    k_ford_validation(x_pca, y_pca, mlp_model)
    print("my model pca:")
    k_ford_validation(x_pca, y_pca, model)
