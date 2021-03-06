import io
import itertools
from datetime import datetime

import sklearn
import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow_core.python import confusion_matrix
from sklearn.metrics import roc_curve

# hyperparams

batch_size = 12
epochs = 1
num_classes = 10


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.002
    if epoch > 5:
        learning_rate = 0.0002
    if epoch > 10:
        learning_rate = 0.0001
    if epoch > 30:
        learning_rate = 0.00005

    # tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = tf.reshape(x_train, (-1, 28, 28, 1))
    x_test = tf.reshape(x_test, (-1, 28, 28, 1))

    # Create validation set
    val_x = x_train[-1000:]
    val_y = y_train[-1000:]
    x_train = x_train[:-1000]
    y_train = y_train[:-1000]

    y_train = tf.one_hot(y_train, num_classes)
    val_y = tf.one_hot(val_y, num_classes)
    # y_test = tf.one_hot(y_test, num_classes)
    # train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ]
    )

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    # Model training
    model.fit(x=x_train,
              y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[lr_callback],
              validation_data=(val_x, val_y),
              verbose=1)

# %%

    y_pred = model.predict_classes(x_test)

    cmat = confusion_matrix(labels=y_test, predictions=y_pred, num_classes=num_classes).numpy()

    plot_confusion_matrix(cmat, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.show()
    # # Model evaluation on test set
    # eval_metrics_list = model.evaluate(x=x_test,
    #                                    y=y_test,
    #                                    verbose=1)
    #
    # print(eval_metrics_list)
