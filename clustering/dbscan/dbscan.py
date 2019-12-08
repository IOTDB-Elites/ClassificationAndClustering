import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from clustering.preprocess import pre_process
from mpl_toolkits.mplot3d import axes3d, Axes3D

def group(full_list):
    num = max(full_list) + 2
    print(num)
    res = []
    for i in range(0, num):
        res.append([])
    for i in range(0, full_list.shape[0]):
        if full_list[i] == -1:
            res[len(res) - 1].append(i)
            # continue;
        else:
            res[full_list[i]].append(i)
    return res


def get_sub_list(full_list, indexes):
    res = []
    for i in indexes:
        res.append(full_list[i])
    return res


if __name__ == '__main__':
    full_data, data_pca, data_respectively_pca = pre_process.load_data("../preprocess/data.csv")

    print(data_respectively_pca)
    print(data_respectively_pca.shape)

    db = DBSCAN(eps=0.1, min_samples=20).fit(data_respectively_pca)
    labels = db.labels_

    grouped = group(labels)

    fig1 = plt.figure()
    ax1 = Axes3D(fig1)

    ax2 = plt.figure().add_subplot(111)
    ax3 = plt.figure().add_subplot(111)
    ax4 = plt.figure().add_subplot(111)

    for i in grouped:
        plot_data = np.array(get_sub_list(data_pca, i))
        ax1.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:,2])
        ax2.scatter(plot_data[:, 0], plot_data[:, 1])
        ax3.scatter(plot_data[:, 0], plot_data[:, 2])
        ax4.scatter(plot_data[:, 1], plot_data[:, 2])

    plt.show()
