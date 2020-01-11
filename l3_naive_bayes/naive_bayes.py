import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from pca.pca import calculate_covariance_matrix, PCA


class NaiveBayes:
    """Clasificator Naiv Bayes"""
    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        self.params = []

    def fit(self, X, y):
        self.X, self.y = X, y
        self.classes = np.unique(y)
        for iter, c in enumerate(self.classes):
            # instantele din X corespunzatoare clasei c
            X_c = X[np.where(y == c)]
            self.params.append([])
            # adaugare mediana si varianta fiecarei proprietati corespunzatoare unei clase
            for col in X_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.params[iter].append(parameters)

    def _calculate_likelihood(self, mean, var, x):
        eps = 1e-4
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exp = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exp

    def _calculate_prior(self, c):
        freq = np.mean(self.y == c)
        return freq

    def _classify(self, sample):
        posteriors = []
        for iter, c in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            for feature_value, params in zip(sample, self.params[iter]):
                likelihood = self._calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._classify(sample) for sample in X]
        return y_pred


# Plot the dataset X and the corresponding labels y in 2D using PCA.
def plot_in_2d(X, y=None, title=None, accuracy=None, legend_labels=None):
    X_transformed = PCA().transform(X, 2)
    x1 = X_transformed[:, 0]
    x2 = X_transformed[:, 1]
    class_distr = []

    y = np.array(y).astype(int)

    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    # Plot the different class distributions
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    # Plot legend
    if not legend_labels is None:
        plt.legend(class_distr, legend_labels, loc=1)

    # Plot title
    if title:
        if accuracy:
            perc = 100 * accuracy
            plt.suptitle(title)
            plt.title("Accuracy: %.1f%%" % perc, fontsize=10)
        else:
            plt.title(title)

    # Axis labels
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.show()


if __name__ == "__main__":
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: ", acc)

    plot_in_2d(X_test, y_pred, accuracy=acc, legend_labels=data.target_names)
