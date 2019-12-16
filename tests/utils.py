import time
import gym
import multiprocessing
from src import Model
from functools import wraps


# Various functions used in unit tests.

def make_model(num_i=4, num_o=4):
    """
    Create empty Model object.

    Args:
        num_i: int, number of inputs.
        num_o: int, number of outputs.

    Returns:
        Model object.
    """
    return Model(num_i, num_o)
