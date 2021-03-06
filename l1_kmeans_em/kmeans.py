"""Summary

Clustering using k-means algorithm

"""

import random
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from sklearn import datasets
from sklearn.cluster import KMeans as KMeansSK

np.random.seed(42)


class KMeans:

    def __init__(self, n_clusters=1, center_init='random', max_iterations=500, tolerance=0.0001):
        """
        :param n_clusters: numarul de clustere
        :param center_init: algoritmul folosit pentru initializarea centrelor
        "random" sau "k-means++"
        :param max_iterations: numarul maxim de iteratii dupa care se opreste modificarea centrelor
        :param tolerance: [procentual] limita sumei diferentelor dintre un nou centru si centrul anterior
        dupa care algoritmul se opreste
        """
        self.n_clusters = n_clusters
        self.center_init = center_init
        self.max_iter = max_iterations
        self.tolerance = tolerance
        # lista centrelor
        self.centroids = []
        # dictionar ce are ca si cheie centrele clusterelor si ca valoare
        # instantele asociate acelor centre
        self.clusters = {}

    def centroids_random_init(self, X):
        """
        Initializare prin alegerea random a centrelor din spatiul instantelor X
        :param X: lista instantelor
        """
        num_samples = len(X)
        for i in range(self.n_clusters):
            rand_int = random.randint(0, num_samples - 1)
            self.centroids.append(X[rand_int])

    def centroids_kmeansplusplus_init(self, X):
        """
        Initializare prin alogoritmul kmeans++
        :param X: lista instantelor
        """
        # primul centru se alege random din instantele X
        num_samples = len(X)
        rand_int = random.randint(0, num_samples - 1)
        self.centroids.append(X[rand_int])
        # Pentru fiecare cluster se calculeaza distanta minima dintre
        # centrul clusterului si fiecare instanta din X (exceptand centrele deja aflate)
        for c in range(self.n_clusters - 1):
            dist = []
            for sample in X:
                if sample not in np.array(self.centroids):
                    distances = [self.euclidean_distance(sample, centroid) for centroid in self.centroids]
                    min_dist = distances.index(min(distances))
                    dist.append(min_dist)
            dist = np.array(dist)
            # Urmatorul centru este instanta cu cea mai mare
            # distanta minima fata de centrul anterior
            next_centroid = X[np.argmax(dist), :]
            self.centroids.append(next_centroid)

    def euclidean_distance(self, p, q):
        # squared_distance = 0
        # for i in range(len(p)):
        #     squared_distance += (p[i] - q[i])**2
        # distance = sqrt(squared_distance)
        return np.linalg.norm(p - q)

    def fit(self, X):
        """
        Asignarea instantelor in clustere si actualizare centre
        :param X: lista instantelor
        """
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
            # Fiecare instanta din X este adaugata in clusterul al carui centru este cel mai apropiat
            for feature_point in X:
                distances = [self.euclidean_distance(feature_point, centroid) for centroid in self.centroids]
                min_dist = distances.index(min(distances))
                self.clusters[min_dist].append(feature_point)

            prev = self.centroids.copy()
            colors = 10 * ['r', 'g', 'b', 'c', 'k']
            for cluster in self.clusters:
                color = colors[cluster]
                for features in self.clusters[cluster]:
                    plt.scatter(features[0], features[1], s=30, color=color)
                    plt.scatter(self.centroids[cluster][0], self.centroids[cluster][1], s=130, color=color,
                                marker='x')
            # Actulizare noi centre
            for cluster_ in self.clusters:
                self.centroids[cluster_] = np.average(self.clusters[cluster_], axis=0)

            isOptimal = True

            # Cat timp numarul de iteratii nu s-a epuizat sau nu a fost atinsa toleranta
            # se actualizeaza centrelex
            for k in range(len(self.centroids)):
                prev_centroid = prev[k]
                current_centroid = self.centroids[k]
                if np.sum((current_centroid - prev_centroid)/prev_centroid * 100.0) >= self.tolerance:
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

    # for cluster in estimator.clusters:
    #     color = colors[cluster]
    #     for features in estimator.clusters[cluster]:
    #         plt.scatter(features[0], features[1], s=30, color=color)
    #         plt.scatter(estimator.centroids[cluster][0], estimator.centroids[cluster][1], s=130, color=color, marker='x')
    #         #plt.scatter(estimator_sklearn.cluster_centers_[cluster][0], estimator_sklearn.cluster_centers_[cluster][1], s=140, marker='o', color=color)
    # plt.show()
    plt.show()
    print("Finish.")
