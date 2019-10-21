import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.layers.core import Dense, Flatten

np.random.seed(100)


def make_abs_dataset(show_plot: bool = False):
    w = np.array([-0.04240011450454, 0.00000000029375, 0.03508217905067, 0.40662691102315])
    p = 2.09945271667129
    a = 0.00025724985785
    lam = np.arange(0, 1, .01).T
    N = len(lam)
    noise = np.divide(np.random.normal(size=N, loc=0, scale=1), 500)
    miu = w[3] * np.power(lam, p) / (a + np.power(lam, p)) + w[2] * np.power(lam, 3) + w[1] * np.power(lam, 2) + w[0] * lam + noise
    if show_plot:
        plt.plot(lam, miu, 'b.-', linewidth=0.2)
        plt.xlabel("lambda - alunecare")
        plt.ylabel("miu - coef de frecare")
        plt.show()
    return miu


class SimpleModel(Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.d1 = Dense(units=18, activation='relu')
        self.d2 = Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        return self.d2(x)


if __name__ == "__main__":
    train_data = make_abs_dataset(show_plot=True)
    model = SimpleModel()
