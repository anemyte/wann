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


def make_model_for_env(env_id):
    """
    Create empty Model object based ob gym environment requirements.

    Args:
        env_id: str, id of the environment.

    Returns:
        Model object.
    """
    tmp_env = gym.make(env_id)
    num_i = tmp_env.observation_space.shape[0]
    if isinstance(tmp_env.action_space, gym.spaces.discrete.Discrete):
        num_o = tmp_env.action_space.n
    else:
        num_o = tmp_env.action_space.shape[0]
    return Model(num_i, num_o)
