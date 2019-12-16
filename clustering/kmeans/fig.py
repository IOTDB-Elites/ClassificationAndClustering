import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import numpy as np

from clustering.preprocess.pre_process import load_data

if __name__ == '__main__':
    raw_data, full_data, data_pca, data_respectively_pca = load_data("../preprocess/data.csv")

    data = data_respectively_pca
    k = 6
    estimator = KMeans(n_clusters=k).fit(data)

    label = estimator.labels_
    sse = estimator.inertia_

    fig1 = plt.figure()
    ax1 = Axes3D(fig1)

    ax2 = plt.figure().add_subplot(111)
    ax3 = plt.figure().add_subplot(111)
    ax4 = plt.figure().add_subplot(111)

    for i in range(k):
        plot_data = data[label == i]
        ax1.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2])
        ax2.scatter(plot_data[:, 0], plot_data[:, 1])
        ax3.scatter(plot_data[:, 0], plot_data[:, 2])
        ax4.scatter(plot_data[:, 1], plot_data[:, 2])

    fig1.savefig('kmeans-3d.png')
