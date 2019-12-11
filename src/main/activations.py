import tensorflow as tf
from pandas import DataFrame


def none(x):
    return x


activations = {
    'none': none,
    'relu': tf.keras.backend.relu,
    'sigmoid': tf.keras.backend.sigmoid,
    'tanh': tf.tanh,
}


def activation_df():
    return DataFrame(columns=list(activations.keys()))
