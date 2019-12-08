import numpy as np


def distance(a, b):
    """
    返回两个向量间的欧式距离
    """
    return np.sqrt(np.sum(np.power(a - b, 2)))


def rand_center(data, k):
    """
    随机设置k个中心点
    """
    m = data.shape[1]  # 数据的维度
    centroids = np.zeros((k, m))
    for j in range(m):
        d_min, d_max = np.min(data[:, j]), np.max(data[:, j])
        centroids[:, j] = d_min + (d_max - d_min) * np.random.rand(k)
    return centroids


def converged(centroids1, centroids2):
    """
    通过判断中心点有没有改变，来决定是否收敛
    """
    set1 = set([tuple(c) for c in centroids1])
    set2 = set([tuple(c) for c in centroids2])
    return set1 == set2


def sse(data, centroids, label):
    n = data.shape[0]
    SSE = np.zeros(n, dtype=np.float)  # 类内凝聚度：簇内误差平方和SSE
    for i in range(n):
        SSE[i] = distance(data[i], centroids[label[i]]) ** 2
    return np.sum(SSE)


def k_means(data, k=2, max_iter=300):
    n = data.shape[0]
    centroids = rand_center(data, k)
    label = np.zeros(n, dtype=np.int)

    finished = False

    count = 0
    while (not finished) and count < max_iter:
        count += 1
        old_centroids = np.copy(centroids)
        for i in range(n):
            min_dist, min_index = np.inf, -1
            for j in range(k):
                dist = distance(data[i], centroids[j])
                if dist < min_dist:
                    min_dist, min_index = dist, j
                    label[i] = j

        for i in range(k):
            k_cluster = data[label == i]
            if len(k_cluster) != 0:
                centroids[i] = np.mean(k_cluster, axis=0)
        finished = converged(old_centroids, centroids)

    return centroids, label, sse(data, centroids, label)