import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import itertools

from pca.pca import PCA


class NaiveBayes:
    """Clasificator Naiv Bayes"""
    def __init__(self):
        """
        X: instantele datasetului
        y: labelurile instantelor
        classes: numele claselor
        params: lista ce contine varianta si mediana corespunzatoare
        instantelor fiecarei clase
        """
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
        """
        Calculare functie de densitate de probabilitate pentru o instanta.
        :param mean: medie
        :param var: varianta
        :param x: instanta
        :return: fdp
        """
        eps = 1e-4
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exp = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exp

    def _calculate_prior(self, c):
        """
        :param c: nume clasa
        :return: frecventa de aparitie a unei clase in dataset
        """
        freq = np.mean(self.y == c)
        return freq

    def _classify(self, sample):
        """
        Calculare apartenenta la o clasa pentru o noua instanta.
        :param sample: instanta de clasificat
        :return: numele clasei pentru care a fost clasificata instanta
        """
        posteriors = []
        for iter, c in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            for feature_value, params in zip(sample, self.params[iter]):
                likelihood = self._calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        Clasificare set de date
        :param X: set de date
        :return: lista predictii
        """
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


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))


if __name__ == "__main__":
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target
    labels = [str(label) for label in data.target_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: ", acc)

    plot_in_2d(X_test, y_pred, accuracy=acc, legend_labels=data.target_names)

    cmm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=data.target_names)

    plot_confusion_matrix(cm=cmm, target_names=data.target_names, title="Naive Bayes Confusion Matrix")

    plt.show()
