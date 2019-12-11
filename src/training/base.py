from src.main.model import Model
from src.main.alterations import AddNode, AddConnection, Alteration
from src.main import nodes
from src.main.utils import IOTable
from src.main.activations import activations
from scipy.special import softmax
from threading import Thread
from collections import Iterable
from datetime import datetime
from pathlib import Path
import multiprocessing
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

        # This hold various data on how many times something was test
        self.io_tests = IOTable()
        self.io_tests_anc = IOTable()
        self.io_scores = IOTable()
        # self.io_tables = [self.io_tests, self.io_tests_anc, self.io_scores]
        self.io_tables = {}

        self.tmp = {0: 0, 1: 0}

    # End of __init__ method
    # -----------------------------------------

    def map_io(self, num_eps=2):
        scores = self.get_io_scores(num_eps=num_eps)
        self._connect_best(scores)

    def get_io_scores(self, input_nodes=None, output_nodes=None, num_eps=2):
        # find best io pairs and connect them
        io_tests = IOTable()
        if input_nodes:
            io_tests.add_nodes(input_nodes, as_input=True, as_output=False)
        else:
            io_tests.add_nodes(self.model.non_output_nodes)
        if output_nodes:
            io_tests.add_nodes(output_nodes, as_input=False, as_output=True)
        else:
            io_tests.add_nodes(self.model.non_input_nodes)
        io_scores = io_tests.copy()

        # plan
        # select a pair of io and test the model
        # repeat while there at least X min tries
        alts = []
        while io_tests.absmin() < num_eps:
            io_pair = self._select_io_pair(io_tests=io_tests, greedy=True)
            alt = self.make_alt(alt_type=self.alts['AddConnection'], kwargs={"inp_node": io_pair[0],
                                                                             "out_node": io_pair[1]})
            io_tests.at[io_pair[1].id, io_pair[0].id] += 1
            alts.append(alt)

        # test all pairs
        self.__run_threads(target=self._test_alt, args_list=alts)
        # collect scores
        for alt in alts:
            io_scores.at[alt['alt'].output_node.id, alt['alt'].input_node.id] += alt['score']

        io_scores = io_scores / io_tests
        return io_scores

    def train(self, num_ep, num_alts):

        alts = self.make_alts(num_alts)
        self.__run_threads(self._test_alt, alts)
        # todo 1. make loop for num_ep; 2. apply best alt
        return alts

    def io_search(self, num_ep=10, tests_per_ep=3, debug=True):
        # create a new node or pick one existing (if possible)
        # to select an existing, see how many times it was tested
        # run get_io_scores for that node and possible inputs

        if self.best_score is None:
            self.recalculate_model_score()

        io_table = self.get_io_table("s_io")

        for n in range(num_ep):

            print(f"Ep: {n} | Best score: {self.best_score}")

            # ------------------
            # Select node

            if self.model.hidden_nodes and np.random.uniform(0, 1) > 0.5:
                # select existing node
                num_uses = io_table.sum(axis=1)
                test_node_id = num_uses[num_uses == num_uses.min()].sample().index[0]
                test_node = self.model.get_node_by_id(test_node_id)
                is_test_node_new = False
            else:
                # create new
                test_node = self._add_node_constructor(no_input=True).new_node
                test_node_id = self.model.add_node(test_node)
                is_test_node_new = True
                self._update_io_tables()

            if debug:
                print('=============================================')
                print(f"New node: {is_test_node_new}, Node id: {test_node_id}, Node name: {test_node.name}")
                if isinstance(test_node, nodes.Linear):
                    print(f"Activation: {test_node.activation_name}, "
                          f"Connected to: {[output.name for output in test_node.outputs]}")

            # -------------
            # Test node

            io_table.loc[test_node.id] += 1
            possible_inputs = self.model.get_possible_inputs(test_node)
            scores = self.get_io_scores(input_nodes=possible_inputs, output_nodes=[test_node], num_eps=tests_per_ep)

            # -------------
            # Evaluate

            if debug:
                print(f"Test results: \n{scores}")

            scores = scores.transpose()
            if scores.values.max() > self.best_score:
                self.best_score = scores.values.max()
                # apply best
                inp_idx = scores.idxmax().values[0]

                if debug:
                    print(f"Connecting nodes: {self.model.get_node_by_id(inp_idx).name} -> "
                          f"{test_node.id} ")

                self.model.connect_by_id(inp_idx, test_node.id)

            else:
                if debug:
                    print(f"Best score {scores.values.max()} < {self.best_score}, reverting changes.")
                # revert and begin next ep
                if is_test_node_new:
                    self.model.remove_node(test_node)
                    self._remove_from_io_tables(test_node)
                    # todo for tomorrow: change existing connections; 1 test per ep is not enough
                # self.recalculate_model_score()

            if debug:
                print('=============================================\n\n')

    def recalculate_model_score(self, num_tests=5):
        graph = self.model.make_graph()
        init, out = self._init_graph(graph)
        scores = []
        for n in range(num_tests):
            score_container = {}
            args_list = [[w, graph, init, out, score_container] for w in self.weights]
            self.__run_threads(target=self._test_graph, args_list=args_list)
            scores.extend(score_container.values())
        self.best_score = self.eval(scores)

    def _test_alt(self, alt):
        scores = {}
        args_list = [[w, alt['graph'], alt['init'], alt['out'], scores] for w in self.weights]
        self.__run_threads(target=self._test_graph, args_list=args_list)
        alt['score'] = self.eval(list(scores.values()))
        return alt['score']

    def _test_graph(self, weight, graph, init, out, score_container=None, render=False, write_logs=False):
        score = 0
        env = gym.make(self.env_id)
        obs = env.reset()
        with tf.compat.v1.Session(graph=graph) as sess:
            if write_logs:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                logdir = str(Path(__file__).parents[2] / 'tests/logs/base/%s') % stamp
                writer = tf.compat.v1.summary.FileWriter(logdir, graph)
            sess.run(init)

            while True:
                if render:
                    env.render()
                action = sess.run(out, feed_dict={"MainInput:0": obs, "sw:0": weight})

                step, reward, is_finished, _ = env.step(action)
                score += reward
                if is_finished:
                    break

            # if write_logs:
            #    sess.run(writer.flush())
        if render:
            env.close()
        if score_container is not None:
            score_container[weight] = score
        return score

    def play(self, num=1, weight=0.5, write_logs=False):
        # play and render
        graph = self.model.make_graph()
        init, out = self._init_graph(graph)
        scores = []

        for _ in range(num):
            scores.append(self._test_graph(weight, graph, init, out, render=True, write_logs=write_logs))
        print(f"Average score: {np.sum(scores)/num}")

    def play_manual(self):
        env = gym.make(self.env_id)
        env.reset()
        score = 0
        while True:
            env.render()
            print(step)
            key = int(input())
            step, reward, is_finished, _ = env.step(key)
            score += reward
            if is_finished:
                break
        env.close()
        print(score)

    # ======================================================
    # SUPPLEMENTARY FUNCTIONS

    def get_io_table(self, table_name):
        # return table if it exists create table if it does not

        if table_name not in self.io_tables:
            table = IOTable()
            self._update_io_table(table)
            self.io_tables[table_name] = table
        else:
            table = self.io_tables[table_name]
        return table

    def make_alt(self, alt_type=None, kwargs=None):
        # alt type is a value of self.alts[alt_type]
        alt = dict()
        if alt_type:
            alt['alt_type'] = alt_type
        else:
            alt['alt_type'] = np.random.choice(list(self.alts.values()))  # change with prob-selection
        if kwargs:
            alt['alt'] = alt['alt_type']['constructor'](**kwargs)
        else:
            alt['alt'] = alt['alt_type']['constructor']()
        alt['graph'] = alt['alt'].make_graph()
        alt['init'], alt['out'] = self._init_graph(alt['graph'])
        alt['score'] = None
        return alt

    def make_alts(self, num_alts, alt_type=None):
        alts = []
        for _ in range(num_alts):
            alts.append(self.make_alt(alt_type))
        return alts

    def eval(self, array):
        # evaluate raw scores obtained during model test
        # returns mean score adjusted by std, thus scores with lesser distribution will have greater value
        # this function intended to be replaced by custom
        mean = np.mean(array)
        return mean
        std = np.std(array)
        if std == 0:
            std = 1  # to avoid possible division by zero
        if mean > 0:
            return mean / std
        else:
            return mean * std

    def __run_threads(self, target, args_list):
        threads = []
        for args in args_list:
            if not isinstance(args, Iterable) or isinstance(args, dict):
                args = [args]
            threads.append(Thread(target=target, args=args))
            threads[-1].start()
        for thread in threads:
            thread.join()

    def _init_graph(self, graph):
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
            # TODO this should be different in case of not discrete action space
            out = tf.argmax(graph.get_tensor_by_name("MainOutput:0"))
        return init, out

    def _update_io_table(self, table):
        table.add_nodes(self.model.nodes)

    def _update_io_tables(self):
        for io_table in self.io_tables.values():
            self._update_io_table(io_table)

    def _remove_from_io_table(self, table, node):
        table.remove_node(node)

    def _remove_from_io_tables(self, node):
        for io_table in self.io_tables.values():
            self._remove_from_io_table(io_table, node)

    def _select_input_node(self, io_table, out_node=None, greedy=False):
        # Select input node based on the table provided
        # If inp node specified, the selection will only respect usage amounts for that node
        # otherwise it'll pick a node based on total number of uses

        # out_node - either id or node itself
        # greedy - boolean, if true the algorithm always picks one of the nodes that was tested less frequently
        #          otherwise, it will select least tested node with higher probability

        if out_node:
            if not isinstance(out_node, nodes.Node):
                out_node = self.model.get_node_by_id(out_node)

            possible_inputs = self.model.get_possible_input_ids(out_node, exclude_existing=True)
            if not possible_inputs:
                possible_inputs = self.model.get_possible_input_ids(out_node, exclude_existing=False)
            input_uses = io_table.loc[out_node.id][possible_inputs]
        else:
            input_uses = io_table.sum(axis=0)

        if greedy:
            inp_id = input_uses[input_uses == input_uses.min()].sample().index.values[0]
        else:
            inp_id = np.random.choice(input_uses.index.values, p=softmax(input_uses.values * -1))

        inp_node = self.model.get_node_by_id(inp_id)
        return inp_node

    def _select_output_node(self, io_table, inp_node=None, greedy=False):
        # Select output node based on the table provided
        # If inp node specified, the selection will only respect usage amounts for that node
        # otherwise it'll pick a node based on total number of uses

        # inp_node - either id or node itself
        # greedy - boolean, if true the algorithm always picks one of the nodes that was tested less frequently
        #          otherwise, it will select least tested node with higher probability

        io_table = io_table.copy()
        if inp_node:
            if isinstance(inp_node, nodes.Node):
                out_node_uses = io_table[inp_node.id]
            else:
                out_node_uses = io_table[inp_node]
        else:
            out_node_uses = io_table.sum(axis=1)

        if greedy:
            out_id = out_node_uses[out_node_uses == out_node_uses.min()].sample().index.values[0]
        else:
            out_id = np.random.choice(out_node_uses.index.values, p=softmax(out_node_uses.values * -1))

        out_node = self.model.get_node_by_id(out_id)
        return out_node

    def _add_node_constructor(self, inp_node=None, out_node=None, greedy=False, no_input=False):
        # Create an alteration object that'll add a node to the model

        table = self.get_io_table('anc_io')

        if out_node:
            if not isinstance(out_node, nodes.Node):
                out_node = self.model.get_node_by_id(out_node)
        else:
            out_node = self._select_output_node(io_table=table, inp_node=inp_node, greedy=greedy)

        if no_input:
            pass
        else:
            if inp_node:
                if not isinstance(inp_node, nodes.Node):
                    inp_node = self.model.get_node_by_id(inp_node)
            else:
                inp_node = self._select_input_node(io_table=table, out_node=out_node, greedy=greedy)

        new_node = nodes.Linear(activation=np.random.choice(list(activations.keys())))
        new_node.outputs.append(out_node)

        if no_input:
            table.loc[out_node.id] += 1
        else:
            new_node.inputs.append(inp_node)
            table.at[out_node.id, inp_node.id] += 1

        alt = AddNode(self.model, new_node)
        return alt

    # End of _add_node_constructor function
    # -----------------------------------------

    def _add_conn_constructor(self, inp_node=None, out_node=None, greedy=False):
        # constructor method for new connection gene

        table = self.get_io_table('acc_io')

        if out_node is not None:
            if not isinstance(out_node, nodes.Node):
                out_node = self.model.get_node_by_id(out_node)
        else:
            out_node = self._select_output_node(io_table=table, inp_node=inp_node, greedy=greedy)

        if inp_node is not None:
            if not isinstance(inp_node, nodes.Node):
                inp_node = self.model.get_node_by_id(inp_node)
        else:
            inp_node = self._select_input_node(io_table=table, out_node=out_node, greedy=greedy)

        alt = AddConnection(self.model, inp_node, out_node)
        table.at[out_node.id, inp_node.id] += 1

        return alt

    def _connect_best(self, io_table):
        table = io_table.copy()
        alts = []
        for idx in table.index.values:
            best = table.loc[idx].idxmax()
            alt = AddConnection(model=self.model,
                                inp_node=self.model.get_node_by_id(best),
                                out_node=self.model.get_node_by_id(idx)
            )
            alt.apply()
            alts.append(alt)
        return alts

    # ======================================================
    # DEPRECATED
    # TODO REMOVE

    def train_old(self, num_ep, num_alts):
        for _ in range(num_ep):
            alts = []
            for n in range(num_alts):
                alt_type = np.random.choice(list(self.alts.values()))  # change with prob-selection
                alt = alt_type['constructor']()
                alts.append(alt)
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

    def test_model(self, model, tests_per_weight=1, weights=None):
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

    def _select_io_pair(self, exclude_existing=False, io_tests=None, greedy=False, inp_node=None, out_node=None):
        # Select a pair of nodes that can be connected
        # The selection is based on amount of times a node was tested
        # Nodes that were tested less frequently are more likely to be selected by this method
        # With greedy as True, the selected nodes are those that were least tested (not likely to be)

        if io_tests is None:
            io_tests = self.io_tests

        # Select out node
        out_node_uses = io_tests.sum(axis=1)
        if greedy:
            out_id = out_node_uses[out_node_uses == out_node_uses.min()].sample().index.values[0]
        else:
            out_id = np.random.choice(out_node_uses.index.values, p=softmax(out_node_uses.values * -1))
        out_node = self.model.get_node_by_id(out_id)

        possible_inputs = self.model.get_possible_input_ids(out_node, exclude_existing)
        if not possible_inputs:
            io_tests.loc[out_id] += 1
            return self._select_io_pair(exclude_existing=exclude_existing, io_tests=io_tests, greedy=greedy)

        # Select input node in the same fashion like output node
        if greedy:
            io_slice = io_tests.loc[out_id][possible_inputs]
            inp_id = io_slice[io_slice == io_slice.min()].sample().index.values[0]
        else:
            inp_id = np.random.choice(possible_inputs, p=softmax(io_tests.loc[out_id][possible_inputs].values * -1))
        inp_node = self.model.get_node_by_id(inp_id)

        return inp_node, out_node

    def _select_new_io_pair(self, io_tests=None, greedy=False):
        return self._select_io_pair(exclude_existing=True, io_tests=io_tests, greedy=greedy)


from src.training import workers as w


class GymAgentV2:

    def __init__(self, env_id, model=None, num_workers=None):

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

        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()

        self.processes = []

    def train(self):

        # launch workers and wait for results
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=w.random, args=[queue, None])
        self.processes.append(p)
        p.start()

        return queue

    def kill_all(self):
        for p in self.processes:
            p.terminate()


if __name__ == '__main__':
    #e = GymAgentV2("LunarLander-v2")
    e = GymAgent("CartPole-v0")
    from src.main.alterations import AlterationV2 as A

    e.map_io()
    alt = A(e.model)

    pass