import tensorflow as tf
from pandas import DataFrame


def none(x):
    return x


@tf.function
def nrelu(x):
    # negative ReLU
    return x if x < 0 else 0.


activations = {
    'none': none,
    'relu': tf.keras.backend.relu,
    'nrelu': nrelu,
    'sigmoid': tf.keras.backend.sigmoid,
    'tanh': tf.tanh,
}


def activation_df():
    return DataFrame(columns=list(activations.keys()))
