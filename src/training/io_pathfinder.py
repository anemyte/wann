import numpy as np
from src.main.alterations import ChangeConnections
import threading
import gym
import tensorflow as tf


def run_test(alt):
    env = gym.make("CartPole-v0")
    graph = alt.create_graph()
    weights = [-2.0, -1.5, -0.5, 0.5, 1.5, 2.0]
    with graph.as_default():
        init = tf.compat.v1.global_variables_initializer()
        out = tf.argmax(graph.get_tensor_by_name("MainOutput:0"))
    for w in weights:
        for _ in range(5):
            obs = env.reset()
            score = 0
            with tf.compat.v1.Session(graph=graph) as sess:
                sess.run(init)
                while True:
                    action = sess.run(out, feed_dict={"MainInput:0": obs, "sw:0": w})
                    step, reward, is_finished, _ = env.step(action)
                    score += reward
                    if is_finished:
                        alt.scores.append(score)
                        break


def find(model):
    scores = np.zeros(shape=(model.num_outputs, model.num_inputs))
    tests = np.zeros(shape=(model.num_outputs, model.num_inputs))
    alterations = []

    while tests.min() <= 10:
        # create connection plan
        # for each output node select one least used input
        scheme = []  # ? tuples out-in
        for out_id in range(model.num_outputs):
            # get id of least tested input for this output
            least_used_inputs = np.argwhere(tests[out_id] == np.min(tests[out_id])).flatten()
            inp_id = np.random.choice(least_used_inputs)
            tests[out_id][inp_id] += 1
            scheme.append((inp_id, out_id))

        actual_ids = [(model.input_nodes[i].id, model.output_nodes[o].id) for i, o in scheme]
        alt = ChangeConnections(model, actual_ids)
        alt.scheme = scheme
        alterations.append(alt)


    threads = []
    for alt in alterations:
        t = threading.Thread(target=run_test, kwargs={'alt': alt})
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
        # wait all threads to finish

    # calculate score per io
    for alt in alterations:
        mean_score = np.mean(alt.scores)
        std = np.std(alt.scores)
        if mean_score > 0:
            score = mean_score / std
        else:
            score = mean_score * std
        # write score in scores
        for i, o in alt.scheme:
            scores[o][i] = scores[o][i] + (score - scores[o][i]) / tests[o][i]

    return scores


def train(model, num_epochs=1):
    for _ in range(num_epochs):
        scores = find(model)
        for oid, o in enumerate(scores):
            i = np.argmax(o)
            i = model.input_nodes[i].id
            oid = model.output_nodes[oid].id
            model.connect_by_id(i, oid)
        print(f"Epoch: {_} | Mean score: {np.mean(scores)}")
    return model


def play(model):
    graph = model.make_graph()
    env = gym.make("CartPole-v0")
    with graph.as_default():
        init = tf.compat.v1.global_variables_initializer()
        out = tf.argmax(graph.get_tensor_by_name("MainOutput:0"))
    obs = env.reset()
    score = 0
    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(init)
        while True:
            env.render()
            action = sess.run(out, feed_dict={"MainInput:0": obs, "sw:0": 0.5})
            step, reward, is_finished, _ = env.step(action)
            score += reward
            if is_finished:
                break
    env.close()
    print(score)


if __name__ == "__main__":
    from src.main.model import Model
    m = Model(4, 2)
