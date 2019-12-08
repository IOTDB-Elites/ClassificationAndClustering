from sklearn.decomposition import PCA
import numpy as np


def preprocess(sample_data):
    """
    将数据所有维度都归一化至0-1之间，避免不同维度的量纲对聚类的影响
    """
    d_min, d_max = np.min(sample_data, axis=0), np.max(sample_data, axis=0)
    sample_data = (sample_data - d_min) / (d_max - d_min)
    return sample_data


def load_data():
    """
    :return: (原始的问卷数据Q1-Q28，未预先分割直接用pca做的结果，预先分割成3类后用pca做的结果)
    """
    data = np.loadtxt("/Users/jackietien/Documents/ClassificationAndClustering/clustering/preprocess/data.csv", dtype=np.float, delimiter=",", skiprows=1)

    data = preprocess(data)

    pca = PCA(n_components=1)

    # 问卷中关于课程内容的部分
    data_class_content = pca.fit_transform(data[:, 5:13])

    # 问卷中关于课程对自己的影响部分
    data_class_expectation = pca.fit_transform(data[:, 13: 17])

    # 问卷中关于授课教师的部分
    data_instructor = pca.fit_transform(data[:, 17:])

    data_respectively_pca = np.hstack((data_class_content, data_class_expectation, data_instructor))

    data_pca = PCA(n_components=3).fit_transform(data[:, 5:])

    return data[5:], data_pca, data_respectively_pca
