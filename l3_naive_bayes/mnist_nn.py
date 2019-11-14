from datetime import datetime

import tensorflow as tf
import numpy as np

#hyperparams
batch_size = 12

class SimpleModel(tf.keras.models.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.input_layer = tf.keras.layers.Dense(input_shape=(28, 28, 1), units=128, activation='relu')
        # self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.dense0 = tf.keras.layers.Dense(128, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.input_layer(x)
        x = self.dense0(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    model = SimpleModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # writer = tf.summary.create_file_writer("\\logs\\tensorboard")
    # tf.summary.trace_on(graph=True, profiler=True)
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)


    EPOCHS = 1

    for epoch in range(EPOCHS):
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    print("Finished training.")
