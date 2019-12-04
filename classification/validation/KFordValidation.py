from time import time
from sklearn.model_selection import KFold
import numpy as np


def k_ford_validation(x, y, model):
    cur_time = time()
    if not hasattr(model, "fit"):
        raise AttributeError("model doesn't have fit method")
    if not hasattr(model, "predict"):
        raise AttributeError("model doesn't have predict method")

    kf = KFold(n_splits=5, random_state=0)
    accuracy, precision_res, recall_res, f1_res, f2_res = 0.0, 0.0, 0.0, 0.0, 0.0

    for train_index, test_index in kf.split(x):
        train_x, train_y = x[train_index], y[train_index]
        test_x, test_y = x[test_index], y[test_index]
        model.fit(train_x, train_y)
        score = model.predict(test_x)
        score = score.reshape((score.shape[0]))

        out = score != test_y
        accuracy += 1 - (np.sum(out) / test_x.shape[0])
        precision_res += precision(score, test_y)
        recall_res += recall(score, test_y)
        f1_res += f1(score, test_y)
        f2_res += f2(score, test_y)

    print("----------------")
    print("Time: ", time() - cur_time)
    print("accuracy: ", accuracy / 5)
    print("precision: ", precision_res / 5)
    print("recall: ", recall_res / 5)
    print("f1: ", f1_res / 5)
    print("f2: ", f2_res / 5)
    print("----------------")


def precision(out, y):
    TP = 0
    FP = 0
    for i in range(out.shape[0]):
        if out[i] == 1 and y[i] == 1:
            TP += 1
        elif out[i] == 1 and y[i] == 0:
            FP += 1
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def recall(out, y):
    TP = 0
    FN = 0
    for i in range(out.shape[0]):
        if out[i] == 1 and y[i] == 1:
            TP += 1
        elif out[i] == 0 and y[i] == 1:
            FN += 1
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def f1(out, y):
    precision_val = precision(out, y)
    recall_val = recall(out, y)

    if precision_val + recall_val == 0:
        return 0

    return 2 * precision_val * recall_val / (precision_val + recall_val)


def f2(out, y):
    precision_val = precision(out, y)
    recall_val = recall(out, y)

    if precision_val + recall_val == 0:
        return 0

    return 5 * precision_val * recall_val / (4 * precision_val + recall_val)
