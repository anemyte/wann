from src.main.model import Model
from src.main.genes import AddNode, AddConnection
from src.main import nodes
from src.main.activations import activations
from threading import Thread
from multiprocessing import Pool, cpu_count
import gym
import numpy as np
import tensorflow as tf


class EvoSearchAgent:

    def __init__(self, env_id, model=None):
        self.env_id = env_id
        self.weights = [-2., -1.5, -0.5, 0.5, 1.5, 2.]

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

        # Future improvements
        self.gene_types = {
            "AddNode": AddNode,
            #"AddConnection": AddConnection
        }
        self._gene_constructors = {
            AddNode: self._ANGConstructor,
            AddConnection: self._ANCConstructor,
        }
        self.gene_avg_score_by_type = {k: {"num_tests": 0, "avg_score": None} for k in self.gene_types}
        gene_score_icr = {}

    def train(self, num_epochs=1, pop_size=5, pop_ttl=5, num_genes=5):
        for i in range(num_epochs):
            pops = self.model.clone_n(pop_size)
            threads = []
            for p in pops:
                threads.append(Thread(target=self._evolve_pop, args=(p, pop_ttl, num_genes)))
                threads[-1].start()
            for t in threads:
                t.join()

            # evolve each pop
            # apply good genes to model
            # repeat

            # TODO replace this temporary solution
            best_pop = max(pops, key=lambda p: p.best_score)
            self.model = best_pop

    def _evolve_pop(self, pop, ttl, num_genes):
        for _ in range(ttl):
            # create a bunch of genes
            genes = self._make_genes(model=pop, num=num_genes)
            # evaluate each of them
            for gene in genes:
                score = self.test_model(gene.test_model)
                gene.score = score

            # apply best gene if it was better than the previous model
            best_gene = max(genes, key=lambda g: g.score)
            if pop.best_score is None or pop.best_score < best_gene.score:
                best_gene.apply()
                pop.best_score = best_gene.score

    def _make_genes(self, model, num):
        genes = []
        # select gene type (Add Node, Add Connection, DeactivateGene)
        for _ in range(num):
            # # evaluate what genes were most profitable
            # # select type with accordance to avg score boost
            # TODO replace with weight-based selection
            gene_type = np.random.choice(list(self.gene_types.values()))

            # select gene properties
            genes.append(self._gene_constructors[gene_type](model))

        return genes

    def _ANGConstructor(self, model):
        # constructor method for new node gene
        # TODO replace random selection with weight-based
        out_node = np.random.choice(model.output_nodes)
        while out_node.inputs and np.random.uniform(0, 1) > 0.5:
            possible_outputs = []
            for node in out_node.inputs:
                if isinstance(node, nodes.Input):
                    continue
                else:
                    possible_outputs.append(node)
            if possible_outputs:
                out_node = np.random.choice(possible_outputs)
            else:
                break
        possible_inputs = model.get_possible_inputs(out_node, exclude_existing=False)
        prev_node = np.random.choice(possible_inputs)
        # TODO replace random with weight-based
        new_node = nodes.Linear(activation=np.random.choice(list(activations.keys())))
        new_node.inputs.append(prev_node)
        new_node.outputs.append(out_node)
        new_gene = self.gene_types['AddNode'](model, new_node)
        return new_gene

    def _ANCConstructor(self, model):
        # constructor method for new connection gene
        pass

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
    e = EvoSearchAgent("CartPole-v0")