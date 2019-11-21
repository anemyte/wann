from src.main.model import Model
from src.main.alterations import AddNode, AddConnection
from src.main import nodes
from src.main.utils import IOTable
from src.main.activations import activations
from scipy.special import softmax
from threading import Thread
from multiprocessing import Pool, cpu_count
import gym
import numpy as np
import tensorflow as tf


class GymAgent:

    def __init__(self, env_id, model=None):
        self.env_id = env_id
        self.weights = [-2., -1.5, -0.5, 0.5, 1.5, 2.]
        self.best_score = None

        # find io dimensions
        temp_env = gym.make(env_id)
        if isinstance(temp_env.action_space, gym.spaces.discrete.Discrete):
            self.num_outputs = temp_env.action_space.n
            self.is_actspace_discrete = True
        else:
            self.num_outputs = temp_env.action_space.shape[0]
            self.is_actspace_discrete = False

        self.num_inputs = temp_env.observation_space.shape[0]

        # init model
        if model is None:
            self.model = Model(self.num_inputs, self.num_outputs)
        else:
            self.model = model

        self.alts = {
            "AddNode": {
                "class": AddNode,
                "constructor": self._add_node_constructor,
                "effectiveness": 0,
                "use_times": 0,
            },
            "AddConnection": {
                "class": AddConnection,
                "constructor": self._add_conn_constructor,
                "effectiveness": 0,
                "use_times": 0,
            },
        }

        # These hold information how many times a pair of nodes was tested and average score per pair
        self.io_tests = IOTable()
        self.io_scores = IOTable()
        self.io_tables = [self.io_tests, self.io_scores]
        self._update_nodes_in_io_tables()

        self.tmp = []

    # End of __init__ method
    # -----------------------------------------

    def _update_nodes_in_io_tables(self):
        for io_table in self.io_tables:
            io_table.add_nodes(self.model.nodes)

    def _select_io_pair(self):
        # Select a pair of nodes that can be connected
        # The selection is based on amount of times a node was tested
        # Nodes that were tested less frequently are more likely to be selected by this method

        # Select out node
        out_node_uses = self.io_tests.sum(axis=1)
        out_id = np.random.choice(out_node_uses.index.values, p=softmax(out_node_uses.values * -1))
        out_node = self.model.get_node_by_id(out_id)

        possible_inputs = self.model.get_possible_input_ids(out_node)
        if not possible_inputs:
            self.io_tests.loc[out_id] += 1
            return self._select_io_pair()

        # Select input node in the same fashion like output node
        inp_id = np.random.choice(possible_inputs, p=softmax(self.io_tests.loc[out_id][possible_inputs].values * -1))
        inp_node = self.model.get_node_by_id(inp_id)

        return inp_node, out_node

    # End of _select_io_pair function
    # -----------------------------------------

    def _add_node_constructor(self):
        # Add Node (alteration) constructor
        # Create an alteration object that'll add a node to the model

        inp_node, out_node = self._select_io_pair()
        new_node = nodes.Linear(activation=np.random.choice(list(activations.keys())))
        new_node.inputs.append(inp_node)
        new_node.outputs.append(out_node)

        alt = AddNode(self.model, new_node)
        self.io_tests.at[out_node.id, inp_node.id] += 1
        self.alts['AddNode']['use_times'] += 1

        return alt

    # End of _add_node_constructor function
    # -----------------------------------------

    def _add_conn_constructor(self):
        # constructor method for new connection gene

        inp_node, out_node = self._select_io_pair()
        alt = AddConnection(self.model, inp_node, out_node)
        self.io_tests.at[out_node.id, inp_node.id] += 1
        self.alts['AddConnection']['use_times'] += 1
        return alt

    def test_model(self, model, tests_per_weight=3, weights=None):
        # model - an instance of model to test in environment
        if weights is None:
            weights = self.weights
        scores = []
        env = gym.make(self.env_id)
        graph = model.make_graph()
        init, out = self._init_graph(graph)
        for w in weights:
            for _ in range(tests_per_weight):
                scores.append(self._test_graph(w, graph, init, out, env))
        return self.eval(scores)

    def _init_graph(self, graph):
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
            # TODO this should be different in case of not discrete action space
            out = tf.argmax(graph.get_tensor_by_name("MainOutput:0"))
        return init, out

    def _test_graph(self, weight, graph, init, out, env, render=False):
        score = 0
        obs = env.reset()
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)
            while True:
                if render:
                    env.render()
                action = sess.run(out, feed_dict={"MainInput:0": obs, "sw:0": weight})
                step, reward, is_finished, _ = env.step(action)
                score += reward
                if is_finished:
                    break
        if render:
            env.close()
        return score

    def train(self, num_ep):
        for _ in range(num_ep):
            alt_type = np.random.choice(list(self.alts.values()))  # change with prob-selection
            alt = alt_type['constructor']()
            self.tmp.append(alt)
            alt.apply_to(alt.test_model)
            score = self.test_model(alt.test_model)
            alt.revert_for(alt.test_model)
            if self.best_score is None:
                self.best_score = score
            elif self.best_score < score:
                alt.apply()
                self._update_nodes_in_io_tables()
            else:
                continue

    def play(self, num=1, weight=0.5, seed=None):
        # play and render
        graph = self.model.make_graph()
        init, out = self._init_graph(graph)
        env = gym.make(self.env_id)
        if seed is not None:
            env.seed(seed)
        for _ in range(num):
            self._test_graph(weight, graph, init, out, env, render=True)

    def eval(self, array):
        # evaluate raw scores obtained during model test
        # returns mean score adjusted by std, thus scores with lesser distribution will have greater value
        # this function intended to be replaced by custom
        mean = np.mean(array)
        std = np.std(array)
        if std == 0:
            std = 1  # to avoid possible division by zero
        if mean > 0:
            return mean / std
        else:
            return mean * std


if __name__ == '__main__':
    e = GymAgent("CartPole-v0")
    it = IOTable()
    it.add_nodes(e.model.nodes)
    lin = nodes.Linear()
    lin.id = 8
    it.add_node(lin)
    it.at[8, 1] = 2
    it.at[4, 3] = 1