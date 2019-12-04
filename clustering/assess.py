from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from clustering import k_means

data = np.loadtxt("data.csv", dtype=np.float, delimiter=",", skiprows=1)

pca = PCA(n_components=10)

data = pca.fit_transform(data)

SSE = []
# 轮廓系数
silhouette = []

for k in range(2, 20):
    best_sse = np.inf
    best_label = None
    # 算法可能局部收敛的问题，随机多运行几次，取最优值
    for i in range(10):
        centroids, label, sse = k_means.k_means(data, k, False)
        if sse < best_sse:
            best_sse = sse
            best_label = label

    print(k, best_sse, silhouette_score(data, best_label))
    SSE.append(best_sse)
    silhouette.append(silhouette_score(data, best_label))

print(SSE)
print(silhouette)