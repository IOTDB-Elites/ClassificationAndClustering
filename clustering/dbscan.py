import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def group(full_list):
    num = max(full_list) + 2
    res = []
    for i in range(0, num):
        res.append([])
    for i in range(0, full_list.shape[0]):
        if full_list[i] == -1:
            res[len(res) - 1].append(i)
            # continue
        else:
            res[full_list[i]].append(i)
    return res


def get_sub_list(full_list, indexes):
    res = []
    for i in indexes:
        res.append(full_list[i])
    return res


if __name__ == '__main__':
    data = np.loadtxt("data.csv", dtype=np.float, delimiter=",", skiprows=1)

    # data1 = data[:, 5:]
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)

    db = DBSCAN(eps=0.5, min_samples=5).fit(data)
    labels = db.labels_

    grouped = group(labels)

    for i in grouped:
        plot_data = np.array(get_sub_list(data, i))
        plt.scatter(plot_data[:, 0], plot_data[:, 1])

    plt.show()
