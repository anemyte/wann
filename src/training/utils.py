import gym
import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime


def test_graph_gym(weight, graph, init, out, env_id, seed=None, render=False,
                   score_container=None, test_id=None):
    """
    Test graph in gym and obtain score.

    :param weight: float value to assign to shared weight.
    :param graph: tf.Graph object.
    :param init: init operation to create variables in the graph.
    :param out: out tensor, result of the graph computation.
    :param env_id: string, gym environment id to create.
    :param seed: int, create specific environment (the way to control randomisation in the seed).
    :param render: bool, whether to display the game process.
    :param score_container: dict, a container where to put the result score.
    :param test_id: any hashable, the key at which to save output in the score container. Default is weight value.
    :return: score, a float value accumulated through the testing.
    """

    score = 0
    env = gym.make(env_id)

    if seed is not None:  # apply seed if passed
        env.seed(seed)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(init)  # init variables
        obs = env.reset()  # get initial observation
        while True:
            if render:
                env.render()
            action = sess.run(out, feed_dict={"MainInput:0": obs, "sw:0": weight})
            obs, reward, is_finished, _ = env.step(action)
            score += reward
            if is_finished:
                break
    sess.close()

    if render:
        env.close()  # this is to avoid game window hanging when the episode is over

    if score_container is not None:
        if test_id is not None:
            score_container[test_id] = score
        else:
            score_container[weight] = score

    return score


def test_graph_gym_m(weights, graph, init, out, env_id, seeds):
    """
    Test graph in gym and obtain score.

    :param weights: iterable of float values to assign to shared weight.
    :param graph: tf.Graph object.
    :param init: init operation to create variables in the graph.
    :param out: out tensor, result of the graph computation.
    :param env_id: string, gym environment id to create.
    :param seeds: iterable, create specific environment (the way to control randomisation in the seed).
    :return: scores, a dict of seed_id: [scores for each weight]
    """
    output = dict()
    env = gym.make(env_id)
    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(init)
        for seed in seeds:  # get results per seed
            env.seed(seed)
            scores = []
            for w in weights:  # get results per weight
                obs = env.reset()
                score = 0
                while True:
                    action = sess.run(out, feed_dict={"MainInput:0": obs, "sw:0": w})
                    obs, reward, is_finished, _ = env.step(action)
                    score += reward
                    if is_finished:
                        scores.append(score)
                        break

            output[seed] = scores
    sess.close()
    return output


def pbexec(**kwargs):
    import multiprocessing
    import queue, time
    q = multiprocessing.Queue()
    args = tuple([q])
    p = multiprocessing.Process(target=tggmq, args=args, kwargs=kwargs)
    p.start()
    out = None
    while not out:
        try:
            out = q.get_nowait()
        except queue.Empty:
            time.sleep(1)
    p.terminate()
    return out


def tggmq(q, **kwargs):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    out = test_graph_gym_m(**kwargs)
    q.put(out)


def export_tensorboard(graph, export_path=''):
    # export logs to analyse graph in tensorboard
    with tf.compat.v1.Session(graph=graph):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not export_path:
            export_path = str(Path(__file__).parents[2] / 'tests/logs/base/%s') % stamp
        tf.compat.v1.summary.FileWriter(export_path, graph)
        return export_path


def init_graph(graph, out_func=''):
    with graph.as_default():
        if out_func == 'argmax':
            out = tf.argmax(graph.get_tensor_by_name("MainOutput:0"))
        else:
            # no additional function
            out = graph.get_tensor_by_name("MainOutput:0")
        init = tf.compat.v1.global_variables_initializer()
    return init, out


def get_seeds(env_id, amount):
    env = gym.make(env_id)
    retry_count = 3
    seeds = []

    while retry_count > 0 and seeds.__len__() < amount:
        new_seed = env.seed()[0]
        while new_seed in seeds:
            if retry_count > 0:
                retry_count -= 1
                new_seed = env.seed()[0]
            else:
                break
        else:
            retry_count = 3
            seeds.append(new_seed)

    return seeds


def eval_meanstd_product(array):
    # todo get rid of this
    return eval_mean(array)
    mean = np.mean(array)
    std = np.std(array)
    if std == 0:
        std = 1  # to avoid possible division by zero
    if mean > 0:
        return mean / std
    else:
        return mean * std


def eval_mean(array):
    return np.mean(array)