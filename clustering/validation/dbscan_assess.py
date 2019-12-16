import time

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from clustering.preprocess.pre_process import load_data

raw_data, data_all, data_pca, data_respectively_pca = load_data("../preprocess/data.csv")

data = data_all

lib_SSE = []
# 轮廓系数
# lib_silhouette = []
# lib_calinski_harabasz = []
#
# self_SSE = []
# self_silhouette = []
# self_calinski_harabasz = []
#
#
# lib_SSE.clear()
# lib_silhouette.clear()
# lib_calinski_harabasz.clear()
# self_SSE.clear()
# self_silhouette.clear()
# self_calinski_harabasz.clear()

plt.figure(figsize=(10, 7.5))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

ns = 4
nbrs = NearestNeighbors(n_neighbors=ns, radius=0).fit(data)
distances, indices = nbrs.kneighbors(data)

# print(distances)
# print(indices)
plt.tick_params(labelsize=15)
distanceDec = sorted(distances[:, ns - 1], reverse=False)
plt.plot(distanceDec)
plt.xlabel('Points Sorted According to Distance of 4th Nearest Neighbor', size=15)
plt.ylabel('4th Nearest Neighbor Distance', size=15)
# plt.show()
# plt.savefig('decide_eps.png')

eps = 0.75
min_samples = 20
start_time = time.time()
estimator = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
end_time = time.time()
consuming_time = end_time - start_time
label = estimator.labels_
# sse = estimator.inertia_
silhouette = silhouette_score(data, label)

calinski_harabasz = calinski_harabasz_score(data, label)
print("\ttime:",consuming_time, "\teps: ", eps, "\tmin_samples", min_samples, "\tsilhouette: ", silhouette, "\tcalinski_harabaz_score: ",
      calinski_harabasz)

#
# for eps in np.arange(0.1, 1.6, 0.1):
#     for min_samples in range(5, 21):
#         # start_time = time.time()
#         estimator = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
#         # end_time = time.time()
#         # consuming_time = end_time - start_time
#         label = estimator.labels_
#         # sse = estimator.inertia_
#         silhouette = silhouette_score(data, label)
#
#         calinski_harabasz = calinski_harabasz_score(data, label)
#         print("\teps: ", eps, "\tmin_samples", min_samples, "\tsilhouette: ", silhouette, "\tcalinski_harabaz_score: ", calinski_harabasz)
