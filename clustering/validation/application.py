import numpy as np
from sklearn.metrics import silhouette_score
from clustering.kmeans import k_means
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from clustering.preprocess.pre_process import load_data
import time
from clustering.kmeans import k_means

raw_data, data_all, data_pca, data_respectively_pca = load_data('/Users/jackietien/Documents/ClassificationAndClustering/clustering/preprocess/data.csv')


data = data_respectively_pca
k = 6
start_time = time.time()
estimator = KMeans(n_clusters=k).fit(data)
# _, label, sse = k_means.k_means(data, k)
end_time = time.time()
consuming_time = end_time - start_time
print(consuming_time)
# cluster_centers = estimator.cluster_centers_
#
label = estimator.labels_
sse = estimator.inertia_
silhouette = silhouette_score(data, label)
calinski_harabasz = calinski_harabasz_score(data, label)
print("lib result: \t", " k: ", k, "\tsse: ", sse, "\tsilhouette: ", silhouette, "\tcalinski_harabaz_score: ", calinski_harabasz)
#
# for i in range(k):
#     cluster_k = raw_data[label == i]
#     a = {}
#     for j in range(len(cluster_k)):
#         key = cluster_k[j][1]
#         if key in a:
#             a[key] += 1
#         else:
#             a[key] = 1
#     for class_num in range(1, 14):
#         if class_num in a:
#             a[class_num] = a[class_num] / len(raw_data[raw_data[:, 1] == class_num])
#
#     items = sorted(a.items(), key=lambda x: x[1], reverse=True)
#
#     rank = np.mean(cluster_k)
#     print(i, rank, items)



# cluster_centers, label, sse = k_means.k_means(data, k)
# silhouette = silhouette_score(data, label)
# calinski_harabasz = calinski_harabasz_score(data, label)
# print("self implement: ", " k: ", k, "\tsse: ", sse, "\tsilhouette: ", silhouette, "\tcalinski_harabaz_score: ", calinski_harabasz)
# for i in range(k):
#     cluster_k = raw_data[label == i]
#     print(i, np.mean(cluster_k))