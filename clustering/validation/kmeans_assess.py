from sklearn.metrics import silhouette_score
from clustering.kmeans import k_means
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import time
import matplotlib.pyplot as plt
from clustering.preprocess.pre_process import load_data


data_all, data_pca, data_respectively_pca = load_data('/Users/jackietien/Documents/ClassificationAndClustering/clustering/preprocess/data.csv')

lib_SSE = []
# 轮廓系数
lib_silhouette = []
lib_calinski_harabasz = []

self_SSE = []
self_silhouette = []
self_calinski_harabasz = []

for data_name, data in (('data_all', data_all), ('data_pca', data_pca), ('data_respectively_pca', data_respectively_pca)):
    lib_SSE.clear()
    lib_silhouette.clear()
    lib_calinski_harabasz.clear()
    self_SSE.clear()
    self_silhouette.clear()
    self_calinski_harabasz.clear()
    print(data_name, ":")
    for k in range(2, 20):
        start_time = time.time()
        estimator = KMeans(n_clusters=k).fit(data)
        end_time = time.time()
        consuming_time = end_time - start_time
        label = estimator.labels_
        sse = estimator.inertia_
        silhouette = silhouette_score(data, label)
        calinski_harabasz = calinski_harabasz_score(data, label)
        print("lib result: \t", " k: ", k, " consuming time: ", consuming_time, "\tsse: ", sse, "\tsilhouette: ", silhouette, "\tcalinski_harabaz_score: ", calinski_harabasz)
        lib_SSE.append(sse)
        lib_silhouette.append(silhouette)
        lib_calinski_harabasz.append(calinski_harabasz)

        start_time = time.time()
        _, label, sse = k_means.k_means(data, k)
        end_time = time.time()
        consuming_time = end_time - start_time
        silhouette = silhouette_score(data, label)
        calinski_harabasz = calinski_harabasz_score(data, label)
        print("self implement: ", " k: ", k, " consuming time: ", consuming_time, "\tsse: ", sse, "\tsilhouette: ", silhouette, "\tcalinski_harabaz_score: ", calinski_harabasz)
        self_SSE.append(sse)
        self_silhouette.append(silhouette)
        self_calinski_harabasz.append(calinski_harabasz)

    print(lib_SSE)
    print(lib_silhouette)
    print(lib_calinski_harabasz)

    print(self_SSE)
    print(self_silhouette)
    print(self_calinski_harabasz)

    X = range(2, 20)
    plt.plot(X, lib_SSE, label='lib result', color='red')
    plt.plot(X, self_SSE, label='self-implement result', color='blue')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('SSE')

    plt.savefig(data_name + '-SSE.png')
    plt.close()



