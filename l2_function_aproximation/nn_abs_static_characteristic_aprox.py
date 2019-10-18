import numpy as np
import matplotlib.pyplot as plt


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
        plt.plot(lam, miu, 'bo')
        plt.xlabel("lambda - alunecare")
        plt.ylabel("miu - coef de frecare")
        plt.show()
    return miu


def make_model():
    pass
    # todo


if __name__ == "__main__":
    train_data = make_abs_dataset(show_plot=True)
    print(train_data)
