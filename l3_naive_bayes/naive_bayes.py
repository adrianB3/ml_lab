import numpy as np
import math

from sklearn import datasets


class NaiveBayes:
    """Clasificator Naiv Bayes"""
    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        self.params = []

    def fit(self, X, y):
        self.X, y = X, y
        self.classes = np.unique(y)
        for iter, c in enumerate(self.classes):
            # instantele din X corespunzatoare clasei c
            X_c = X[np.where(y=c)]
            


if __name__ == "__main__":
    data = datasets.load_digits()
