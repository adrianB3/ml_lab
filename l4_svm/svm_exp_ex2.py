"""
SVM experiments over
linear and RBF kernel functions
and C penalty parameter
Ex.2 Lab IA SVM
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

if __name__ == "__main__":

    data = pd.read_csv(Path("../data/wdbc.data"), header=None)

    features = data.loc[:, [2, 3]]
    targets = data.loc[:, 1]

    scatter1 = plt.scatter(features.loc[:, 2], features.loc[:, 3], s=data.loc[:, 5] / 6, alpha=0.2, c=targets, cmap='viridis')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

    # TODO - cross validation check

    kernels = ['rbf', 'linear']
    cs = np.logspace(-1, 3, 100)
    scores1 = []
    scores2 = []

    for kernel_ in kernels:
        for c_ in cs:
            print(c_)
            clf = svm.SVC(gamma='scale', kernel=kernel_, C=c_)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            score = accuracy_score(y_test, y_pred)
            if kernel_ == 'rbf':
                scores1.append(score)
            else:
                scores2.append(score)

    plt.figure()
    plt.plot(cs, scores1, 'b-', cs, scores2, 'r-')

    plt.show()
    print("Finished")
