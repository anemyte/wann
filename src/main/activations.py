import tensorflow as tf


def none(x):
    return x


activations = {
    '': none,
    'none': none,
    'relu': tf.keras.backend.relu,
    'sigmoid': tf.keras.backend.sigmoid,
    'tanh': tf.tanh,
}
