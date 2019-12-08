from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from clustering.kmeans import k_means
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt


def preprocess(sample_data):
    """
    将数据所有维度都归一化至0-1之间，避免不同维度的量纲对聚类的影响
    """
    d_min, d_max = np.min(sample_data, axis=0), np.max(sample_data, axis=0)
    sample_data = (sample_data - d_min) / (d_max - d_min)
    return sample_data


data = np.loadtxt("data.csv", dtype=np.float, delimiter=",", skiprows=1)


data = preprocess(data)

pca = PCA(n_components=1)

data = pca.fit_transform(data)

lib_SSE = []
# 轮廓系数
lib_silhouette = []

self_SSE = []
self_silhouette = []


for k in range(2, 20):
    start_time = time.time()
    estimator = KMeans(n_clusters=k).fit(data)
    end_time = time.time()
    consuming_time = end_time - start_time
    label = estimator.labels_
    sse = estimator.inertia_
    silhouette = silhouette_score(data, label)
    print("lib result: ", " k: ", k, "consuming time: ", consuming_time, " sse: ", sse, " silhouette: ", silhouette)
    lib_SSE.append(sse)
    lib_silhouette.append(silhouette)

    start_time = time.time()
    _, label, sse = k_means.k_means(data, k)
    end_time = time.time()
    consuming_time = end_time - start_time
    silhouette = silhouette_score(data, label)
    print("self implement: ", " k: ", k, "consuming time: ", consuming_time, " sse: ", sse, " silhouette: ", silhouette)
    self_SSE.append(sse)
    self_silhouette.append(silhouette)

print(lib_SSE)
print(lib_silhouette)

print(self_SSE)
print(self_silhouette)

X = range(2, 20)
plt.plot(X, lib_SSE, label='lib result', color='red')
plt.plot(X, self_SSE, label='self-implement result', color='blue')

plt.xlabel('K')
plt.ylabel('SSE')

plt.savefig("SSE.png")

