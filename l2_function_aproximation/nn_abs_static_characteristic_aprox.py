import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

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


class SimpleModel(tf.keras.models.Model):
    def __init__(self, input_size=2, output_size=2):
        super(SimpleModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(units=input_size, activation='relu')
        self.d2 = tf.keras.layers.Dense(units=10, activation='relu')
        self.d3 = tf.keras.layers.Dense(units=output_size, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


@tf.function
def train_step(data, labels, model, loss_obj, optim):
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


if __name__ == "__main__":
    train_data = make_abs_dataset(show_plot=True)
    model = SimpleModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

