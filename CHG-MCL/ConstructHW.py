import math
import numpy as np
import hypergraph_construct_KNN
import hypergraph_construct_kmeans
import hypergraph_construct_KMedoids
import hypergraph_construct_DBSCAN
from sklearn.cluster import DBSCAN


# 关联矩阵的变体：加权关联矩阵Hm ，在H(i,j) 位置上存储一个权重（如 tf-idf 权重、互信息等）。

def constructHW_knn(X,K_neigs,is_probH):  # K_neigs：KNN 选择的 K 个最近邻，即每个样本点连接多少个邻居。is_probH：布尔值，表示是否使用概率超图。

    """incidence matrix"""
    # 通过 KNN 计算超图的关联矩阵 H，关联矩阵 H 表示超图结构，即哪些样本点属于同一个超边（hyperedge）。
    H = hypergraph_construct_KNN.construct_H_with_KNN(X,K_neigs,is_probH)

    # 通过超图的关联矩阵 H 生成图 G
    G = hypergraph_construct_KNN._generate_G_from_H(H)

    return G

def constructHW_kmean(X,clusters):   # clusters：KMeans 聚类时的 簇（cluster）数量。

    """incidence matrix"""
    H = hypergraph_construct_kmeans.construct_H_with_Kmeans(X,clusters)   # 这里的超边（hyperedge）是由 聚类得到的簇 构成的。

    G = hypergraph_construct_kmeans._generate_G_from_H(H)

    return G


def constructHW_dbscan(X, eps, min_samples, is_probH):
    """基于DBSCAN的超图构建"""

    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)  # 得到聚类标签

    # 计算超图的关联矩阵
    H = hypergraph_construct_DBSCAN.construct_H_with_DBSCAN(X, eps, min_samples, is_probH)

    # 从超图的关联矩阵生成图
    G = hypergraph_construct_DBSCAN._generate_G_from_H(H)

    return G


def constructHW_kmedoids(X, n_clusters, is_probH=False):
    H = hypergraph_construct_KMedoids.construct_H_with_kmedoids(X, n_clusters=n_clusters, is_probH=is_probH)
    G = hypergraph_construct_KMedoids.generate_G_from_H(H)
    return G
