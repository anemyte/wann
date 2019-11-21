from src.training.random import RandomSearch
from src import WANN
import gym
import threading
import tensorflow as tf
import numpy as np
import time


class GymAgent:

    def __init__(self, env_id, num_workers=8):
        self.env_id = env_id
        self.num_workers = num_workers

        temp_env = gym.make(env_id)
        num_inputs = temp_env.observation_space.shape[0]
        if temp_env.action_space.shape:
            num_outputs = temp_env.action_space.shape[0]
        else:
            num_outputs = temp_env.action_space.n
        self.w = WANN(num_inputs, num_outputs)
        self.alg = RandomSearch(self.w)

        self.weights = [-2.0, -1.5, -0.5, 0.5, 1.5, 2.0]

    def train(self, num_epochs=10, num_species=10, num_tests=10):
        start = time.perf_counter()
        for ep in range(1, num_epochs+1):
            species = []
            for n in range(num_species):
                species.append(self.alg.new_node())
            threads = []
            for s in species:
                env = gym.make(self.env_id)
                thread = threading.Thread(target=self.run_test,
                                          kwargs={'env': env,
                                                  'specimen': s,
                                                  'repeat': num_tests,
                })
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            scores = []
            for s in species:
                std = np.std(s.scores)
                mean = np.mean(s.scores)
                if mean > 0:
                    scores.append(mean / std)
                else:
                    scores.append(mean * std)
            best = np.argmax(scores)
            species[best].apply()
            print(f"Epoch: {ep} | Played: {ep*num_species*num_tests*self.weights.__len__()} times | "
                  f"Best score: {np.round(scores[best], 2)} | Time passed: {np.round(time.perf_counter()-start, 2)}")


    def run_test(self, env, specimen, repeat):
        graph = specimen.create_graph()
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
            out = tf.argmax(graph.get_tensor_by_name("MainOutput:0"))
        for w in self.weights:
            for _ in range(repeat):
                obs = env.reset()
                score = 0
                with tf.compat.v1.Session(graph=graph) as sess:
                    sess.run(init)
                    while True:
                        action = sess.run(out, feed_dict={"MainInput:0": obs, "sw:0": w})
                        step, reward, is_finished, _ = env.step(action)
                        score += reward
                        if is_finished:
                            specimen.scores.append(score)
                            break
        return specimen

    def play(self, weight=1.5):
        env = gym.make(self.env_id)
        obs = env.reset()
        graph = self.w.model.make_graph()
        score = 0
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
            out = tf.argmax(graph.get_tensor_by_name("MainOutput:0"))
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)
            while True:
                env.render()
                action = sess.run(out, feed_dict={"MainInput:0": obs, "sw:0": weight})
                step, reward, is_finished, _ = env.step(action)
                score += reward
                if is_finished:
                    break
            env.close()
        return score


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    a = GymAgent("CartPole-v0")
    a.train(10,10,4)