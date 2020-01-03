"""Summary

Clustering using k-means algorithm

"""
from math import sqrt
import random

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

from sklearn import datasets
from sklearn.cluster import KMeans as KMeansSK
np.random.seed(42)


class KMeans:

    def __init__(self, n_clusters=1, center_init='random', max_iterations=500, tolerance=0.0001):
        self.n_clusters = n_clusters
        self.center_init = center_init
        self.max_iter = max_iterations
        self.tolerance = tolerance
        self.centroids = []
        self.clusters = {}

    def centroids_random_init(self, X):
        num_samples = len(X)
        for i in range(self.n_clusters):
            rand_int = random.randint(0, num_samples - 1)
            self.centroids.append(X[rand_int])

    def centroids_kmeansplusplus_init(self, X):
        num_samples = len(X)
        rand_int = random.randint(0, num_samples - 1)
        self.centroids.append(X[rand_int])
        for c in range(self.n_clusters - 1):
            dist = []
            for sample in X:
                if sample not in np.array(self.centroids):
                    distances = [self.euclidean_distance(sample, centroid) for centroid in self.centroids]
                    min_dist = distances.index(min(distances))
                    dist.append(min_dist)
            dist = np.array(dist)
            next_centroid = X[np.argmax(dist), :]
            self.centroids.append(next_centroid)

    def euclidean_distance(self, p, q):
        squared_distance = 0
        for i in range(len(p)):
            squared_distance += (p[i] - q[i])**2
        distance = sqrt(squared_distance)
        return distance

    def fit(self, X):

        if self.center_init == 'random':
            self.centroids_random_init(X)
        elif self.center_init == 'k-means++':
            self.centroids_kmeansplusplus_init(X)
        else:
            print("No init algo selected")
            return

        for i in range(self.max_iter):
            self.clusters = {}
            for j in range(self.n_clusters):
                self.clusters[j] = []

            for feature_point in X:
                distances = [self.euclidean_distance(feature_point, centroid) for centroid in self.centroids]
                min_dist = distances.index(min(distances))
                self.clusters[min_dist].append(feature_point)

            prev = self.centroids.copy()
            for cluster_ in self.clusters:
                self.centroids[cluster_] = np.average(self.clusters[cluster_], axis=0)

            isOptimal = True

            for k in range(len(self.centroids)):
                prev_centroid = prev[k]
                current_centroid = self.centroids[k]
                if np.sum((current_centroid - prev_centroid)/prev_centroid * 100.0) > self.tolerance:
                    isOptimal = False

            if isOptimal:
                break


if __name__ == "__main__":

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x = x[:, 0:2]

    estimator = KMeans(n_clusters=3, center_init='k-means++')
    estimator.fit(x)

    estimator_sklearn = KMeansSK(n_clusters=3, init='k-means++')
    estimator_sklearn.fit(x)

    colors = 10*['r', 'g', 'b', 'c', 'k']

    for cluster in estimator.clusters:
        color = colors[cluster]
        for features in estimator.clusters[cluster]:
            plt.scatter(features[0], features[1], s=30, color=color)
            plt.scatter(estimator.centroids[cluster][0], estimator.centroids[cluster][1], s=130, color=color, marker='x')
            plt.scatter(estimator_sklearn.cluster_centers_[cluster][0], estimator_sklearn.cluster_centers_[cluster][1], s=140, marker='o', color=color)
    plt.show()
    print("Finish.")
