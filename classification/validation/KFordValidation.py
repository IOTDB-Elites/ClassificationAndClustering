from sklearn.model_selection import KFold
import numpy as np


def k_ford_validation(x, y, model):
    if not hasattr(model, "fit"):
        raise AttributeError("model doesn't have fit method")
    if not hasattr(model, "predict"):
        raise AttributeError("model doesn't have predict method")

    kf = KFold(n_splits=5, random_state=0)
    for train_index, test_index in kf.split(x):
        train_x, train_y = x[train_index], y[train_index]
        test_x, test_y = x[test_index], y[test_index]
        model.fit(train_x, train_y)
        score = model.predict(test_x)
        score = score.reshape((score.shape[0]))

        out = score != test_y
        print("accuracy: ", 1 - (np.sum(out) / test_x.shape[0]))
