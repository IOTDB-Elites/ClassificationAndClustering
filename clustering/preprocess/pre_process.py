from sklearn.decomposition import PCA
import numpy as np


def load_data(data_path):
    """
    :return: (所有的原始数据，原始的问卷数据Q1-Q28，未预先分割直接用pca做的结果，预先分割成3类后用pca做的结果)
    """
    data = np.loadtxt(data_path, dtype=np.float, delimiter=",", skiprows=1)

    pca = PCA(n_components=1)

    # 问卷中关于课程内容的部分
    data_class_content = pca.fit_transform(data[:, 5:13])

    # 问卷中关于课程对自己的影响部分
    data_class_expectation = pca.fit_transform(data[:, 13: 17])

    # 问卷中关于授课教师的部分
    data_instructor = pca.fit_transform(data[:, 17:])

    data_respectively_pca = np.hstack((data_class_content, data_class_expectation, data_instructor))

    data_pca = PCA(n_components=3).fit_transform(data[:, 5:])

    return data, data[5:], data_pca, data_respectively_pca
