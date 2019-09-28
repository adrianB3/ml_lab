"""Summary

Clustering using Expectation-Maximization algorithm
(Gaussian Mixture Models - GMMs)
Nice tutorial: https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
sklearn docs on gmm:
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
https://scikit-learn.org/stable/modules/mixture.html#gmm
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('fivethirtyeight')

from scipy.stats import multivariate_normal

from sklearn import datasets
from sklearn import mixture

np.random.seed(0)

if __name__ == "__main__":

    wdbc = datasets.load_breast_cancer()
    X = wdbc.data
    y = wdbc.target

    thin_X = X[:, 0:2]

    x, y = np.meshgrid(np.sort(X[:, 0]), np.sort(X[:, 1]))
    XY = np.array([x.flatten(), y.flatten()]).T

    gmm = mixture.GaussianMixture(n_components=2,
                                  covariance_type='full')
    gmm.fit(thin_X)

    print('Converged:', gmm.converged_)  # Check if the model has converged
    means = gmm.means_
    covariances = gmm.covariances_
    print('Means', means)
    print('Covariances', covariances)

    # display predicted scores by the model as a contour plot

    fig = plt.figure(figsize=(10, 10))
    for m, c in zip(means, covariances):
        multi_normal = multivariate_normal(mean=m, cov=c)
        plt.contour(np.sort(thin_X[:, 0]), np.sort(thin_X[:, 1]), multi_normal.pdf(XY).reshape(len(thin_X), len(thin_X)), colors='black',
                    alpha=0.3)
        plt.scatter(m[0], m[1], c='grey', zorder=10, s=100)

    plt.scatter(thin_X[:, 0], thin_X[:, 1])

    plt.title('Clustering with EM-GMM')
    plt.axis('tight')
    plt.show()

    print("Finish")
