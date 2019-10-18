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


class GMM:

    def __init__(self, X, n_components, iterations):
        self.iterations = iterations
        self.n_components = n_components
        self.X = X
        self.miu = None
        self.pic = None
        self.cov = None
        self.XY = None

    def run(self):
        self.reg_cov = 1e-6 * np.identity(len(self.X[0]))
        x, y = np.meshgrid(np.sort(self.X[:, 0]), np.sort(self.X[:, 1]))
        self.XY = np.array([x.flatten(), y.flatten()]).T

        """Init"""
        # set initial miu, covariance and pic values
        # n x M matrix -- n gaussians with m dimensions
        self.miu = np.random.randint(min(self.X[:, 0]), max(self.X[:, 0]), size=(self.n_components, len(self.X[0])))
        self.cov = np.zeros((self.n_components, len(self.X[0]), len(self.X[0])))
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim], 5)
        self.pic = np.ones(self.n_components) / self.n_components
        log_likelihoods = []  # log likelihoods per iteration

        plot_gmm(self.X, self.miu, self.cov, "Initial Step")

        for i in range(self.iterations):
            """E step"""
            r_ic = np.zeros((len(self.X), len(self.cov)))
            for m, co, p, r in zip(self.miu, self.cov, self.pic, range(len(r_ic[0]))):
                co += self.reg_cov
                mn = multivariate_normal(mean=m, cov=co)
                r_ic[:, r] = p * mn.pdf(self.X) / np.sum([pi_c * multivariate_normal(mean=mu_c, cov=cov_c).pdf(self.X)
                                                          for pi_c, mu_c, cov_c in
                                                          zip(self.pic, self.miu, self.cov + self.reg_cov)], axis=0)

            """M step"""
            self.miu = []
            self.cov = []
            self.pic = []
            log_likelihood = []

            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:, c], axis=0)
                mu_c = (1 / m_c) * np.sum(self.X * r_ic[:, c].reshape(len(self.X), 1), axis=0)
                self.miu.append(mu_c)
                self.cov.append(((1 / m_c) * np.dot((np.array(r_ic[:, c]).reshape(len(self.X), 1) * (self.X - mu_c)).T,
                                                    (self.X - mu_c))) + self.reg_cov)

                self.pic.append(m_c / np.sum(r_ic))

            log_likelihoods.append(np.log(np.sum([k * multivariate_normal(self.miu[i], self.cov[j]).pdf(self.X)
                                                  for k, i, j in zip(self.pic, range(len(self.miu)), range(len(self.cov)))])))

        plot_gmm(self.X, self.miu, self.cov, "Final step")


def plot_gmm(X, means, covariances, title):
    x, y = np.meshgrid(np.sort(X[:, 0]), np.sort(X[:, 1]))
    XY = np.array([x.flatten(), y.flatten()]).T

    fig = plt.figure(figsize=(10, 10))
    for m, c in zip(means, covariances):
        multi_normal = multivariate_normal(mean=m, cov=c)
        plt.contour(np.sort(X[:, 0]), np.sort(X[:, 1]),
                    multi_normal.pdf(XY).reshape(len(thin_X), len(thin_X)), colors='black',
                    alpha=0.3)
        plt.scatter(m[0], m[1], c='grey', zorder=10, s=100)

    plt.scatter(thin_X[:, 0], thin_X[:, 1])

    plt.title(title)
    plt.axis('tight')
    # plt.show()


if __name__ == "__main__":
    wdbc = datasets.load_breast_cancer()
    X_data = wdbc.data
    y = wdbc.target

    thin_X = X_data[:, 0:2]

    gmm_own = GMM(X=thin_X, n_components=2, iterations=50)
    gmm_own.run()

    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', max_iter=50)
    gmm.fit(thin_X)

    print('Converged:', gmm.converged_)  # Check if the model has converged
    means_ = gmm.means_
    covariances_ = gmm.covariances_
    print('Means', means_)
    print('Covariances', covariances_)

    # display predicted scores by the model as a contour plot
    plot_gmm(thin_X, means_, covariances_, "sklearn final")
    plt.show()
    print("Finish")
