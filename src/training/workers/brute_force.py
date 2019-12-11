from src.training.workers.base import Worker
from src.main.alterations import AlterationV2
from src.main.activations import activation_df
from src.main.utils import IOTable
from src.main import nodes
from src.training.utils import test_graph_gym, init_graph, test_graph_gym_m, get_seeds, eval_meanstd_product
from scipy.special import softmax
from collections import defaultdict
import numpy as np


class BruteForceWorker(Worker):

    def __init__(self, out_queue, exit_signal, data):
        super(BruteForceWorker, self).__init__(out_queue=out_queue,
                                               exit_signal=exit_signal,
                                               data=data)

        self.weights = self.data['weights']
        self.production_counter = 0
        self.io_tables = defaultdict(self.__io_table_factory)

        self.mod_funcs = []
        prop_list = dir(self)
        for prop in prop_list:
            if prop.startswith("mod_func_"):
                mf = dict()
                mf['id'] = self.mod_funcs.__len__()
                mf['score'] = 0
                mf['num_tests'] = 0
                mf['name'] = prop
                mf['callable'] = eval(f"self.{prop}")
                mf['enabled'] = True
                self.mod_funcs.append(mf)

        self.__activation_uses = activation_df()

    def __io_table_factory(self):
        tbl = IOTable()
        tbl.add_nodes(self.model.nodes)
        return tbl

    def pre_job(self):
        # ===================================================================
        # Change_conn method block
        # Create special tables for 'change_conn' method
        self.io_tables['change_conn_0'] = self.model.get_adjacency_matrix()
        self.io_tables['change_conn_0'].replace(0, np.nan, inplace=True)
        self.io_tables['change_conn_0'].dropna(how='all', inplace=True)
        if self.io_tables['change_conn_0'].empty:
            self.disable_mod_func('mod_func_change_conn')
        else:
            tmp = self.model.get_adjacency_matrix()
            tmp = tmp.loc[self.io_tables['change_conn_0'].index]
            tmp.replace(1, np.nan, inplace=True)
            self.io_tables['change_conn_1'] = tmp
        # End of change_conn block
        # --------------------------------------------------------------------

    def job(self):
        # something to do before the main loop begins
        self.pre_job()
        # main loop
        while not self.stop.value:
            mf = self.select_mod_func()
            alt = mf['callable']()
            success, scores = self.test_alt(alt)
            mf['num_tests'] += 1
            mf['score'] += scores['avg']
            if success:
                # stop all workers
                self.stop.value = 1
                alt.score_data = scores
                self.oq.put(alt)
            else:
                continue
        # exit sequence
        self.post_job()

    def post_job(self, *args, **kwargs):
        # something to do before shutting down
        # use in child classes
        pass

    def select_output(self, table, for_input=None, greedy=False, increment=False):
        # table is IOTable or its id
        if not isinstance(table, IOTable):
            table = self.io_tables[table]

        if for_input is None:
            num_uses = table.sum(axis=1)
        else:
            num_uses = table[for_input]

        num_uses.dropna(inplace=True)

        if greedy:
            result = num_uses[num_uses == num_uses.min()].sample()
        else:
            weights = softmax(num_uses * -1)
            result = num_uses.sample(weights=weights)

        try:  # In case of a very limited pool result might be just a value and not a pd.Series object.
            result = result.index[0]
        except AttributeError:
            pass

        # set record that this pair was tested one more time
        if increment:
            if not for_input:
                raise ValueError("Cannot increment if no 'for_input' value provided")
            table.at[result, for_input] += 1

        return result

    def select_input(self, table, for_output=None, greedy=False, increment=False):
        # table is IOTable or its id
        if not isinstance(table, IOTable):
            table = self.io_tables[table]

        if for_output is None:
            num_uses = table.sum(axis=0)
        else:
            num_uses = table.loc[for_output]

        num_uses.dropna(inplace=True)  # np.nan, if present, ruin softmax

        if greedy:
            result = num_uses[num_uses == num_uses.min()].sample()
        else:
            weights = softmax(num_uses * -1)
            result = num_uses.sample(weights=weights)

        try:  # In case of very limited pool result might be just a value and not a pd.Series object.
            result = result.index[0]
        except AttributeError:
            pass

        # set record that this pair was tested one more time
        if increment:
            if not for_output:
                raise ValueError("Cannot increment if no 'for_output' value provided")
            table.at[for_output, result] += 1
        return result

    def select_activation(self, idx, greedy=False, increment=True):
        if idx not in self.__activation_uses.index:
            self.__activation_uses.loc[idx] = 0

        slice = self.__activation_uses.loc[idx]

        if greedy:
            result = slice[slice == slice.min()].sample().index[0]
        else:
            weights = softmax(slice.astype('int') * -1)
            result = slice.sample(weights=weights).index[0]

        if increment:
            self.__activation_uses.at[idx, result] += 1

        return result

    def select_mod_func(self):
        scores = [None] * self.mod_funcs.__len__()
        for mf in self.mod_funcs:
            try:
                if mf['enabled']:
                    if mf['num_tests'] == 0:
                        # force it to be used at least once
                        return mf
                    scores[mf['id']] = mf['score'] / mf['num_tests']
                else:
                    scores[mf['id']] = -np.inf  # disabled method will have zero probability to be selected
            except ZeroDivisionError:
                scores[mf['id']] = 0

        probs = softmax(scores)
        choice = np.random.choice(list(range(probs.__len__())), p=probs)
        return self.mod_funcs[choice]

    def test_alt(self, alt):
        # run a set of increasingly difficult tests
        graph = alt.make_graph()
        init, out = init_graph(graph, self.data['out_func'])
        # test 1
        # score_l1 = self.test_graph_l1(graph=graph, init=init, out=out)
        # if score_l1 < self.data['target_score']:
        #    return False, {'avg': score_l1}
        # test 2
        scores_l2 = self.test_graph_l2(graph=graph, init=init, out=out)
        # calculate average
        avg_l2 = eval_meanstd_product(scores_l2[self.data['env_seed']])
        if avg_l2 <= self.data['target_score']:
            return False, {'avg': avg_l2}
        # test 3 (final)
        scores_l3 = self.test_graph_l3(graph=graph, init=init, out=out)
        # calculate average
        avg_l3 = eval_meanstd_product(list(scores_l3.values()))
        # compare against model's best score
        if avg_l3 <= self.data['best_score']:
            return False, {'avg': avg_l3}
        else:
            # prepare return object
            scores_l3['l2_avg'] = avg_l2
            scores_l3['avg'] = avg_l3
            return True, scores_l3

    def test_graph_l1(self, graph, init, out):
        # very simple one-run test
        # low quality, high probability of 'lucky' pass
        random_weight = np.random.choice(self.weights)
        score = test_graph_gym(weight=random_weight, graph=graph, init=init, out=out,
                               env_id=self.data['env_id'], seed=self.data['env_seed'])
        return score

    def test_graph_l2(self, graph, init, out):
        # more advanced than l1, run several times, once with each weight
        # less than l1 but still significant chance to pass by luck
        scores = test_graph_gym_m(weights=self.weights, graph=graph, init=init, out=out,
                                  env_id=self.data['env_id'], seeds=[self.data['env_seed']])
        return scores

    def test_graph_l3(self, graph, init, out):
        # advanced and most time-consuming test
        # the graph is tested with random seeds and per each seed the test is performed with each of weights

        seeds = get_seeds(env_id=self.data['env_id'], amount=5)
        scores = test_graph_gym_m(weights=self.weights, graph=graph, init=init, out=out,
                                  env_id=self.data['env_id'], seeds=seeds)
        return scores

    def _make_empty_alt(self):
        self.production_counter += 1
        # Initialise new alteration and add some information for performance analysis
        new_alt = AlterationV2(self.model)
        new_alt.wid = self.getName().split('-')[1]
        new_alt.wname = self.name
        new_alt.pid = self.production_counter
        return new_alt

    def disable_mod_func(self, method_name):
        for mf in self.mod_funcs:
            if mf['name'] == method_name:
                mf['enabled'] = False
                break
        else:
            raise ValueError(f"No method {method_name} found.")

    def mod_func_add_conn(self):
        alt = self._make_empty_alt()

        # select two nodes to connect
        out_node_id = self.select_output(table='add_conn_0')
        in_node_id = self.select_input(table='add_conn_0', for_output=out_node_id, increment=True)
        # connect 'em
        alt.add_connection(in_node_id, out_node_id)
        return alt

    def mod_func_change_conn(self):
        # ========================================================================
        # Self-check block
        # Check that the method can be used and that there is some sense to use it
        # If any of these checks evaluates to True, the method will be disabled and
        # another mod method will be called to create an alteration.
        # ------------------------------------------------------------------------
        # Check 1: Model has at least one connection.
        check_1 = self.io_tables['change_conn_0'].get_connections().__len__() == 0
        if check_1:
            self.disable_mod_func('mod_func_change_conn')
            return self.mod_func_add_conn()
        # Check 2: There are some untested input-output pairs left.
        check_2 = False
        tmp = self.io_tables['change_conn_0'].sum(axis=1).dropna() > 0
        for idx in tmp.index:
            if 0 in self.io_tables['change_conn_1'].loc[idx].values:
                break
            else:
                continue
        else:
            check_2 = True
        if check_2:
            self.disable_mod_func('mod_func_change_conn')
            return self.mod_func_add_conn()
        # End of self-test block
        # ------------------------------------------------------------------------

        alt = self._make_empty_alt()
        out_node_id = self.select_output(table='change_conn_0')
        old_input_id = self.select_input(table='change_conn_0', for_output=out_node_id, increment=True)
        new_inp_id = self.select_input(table='change_conn_1', for_output=out_node_id, increment=True)

        # prepare alteration
        alt.remove_connection(old_input_id, out_node_id)
        alt.add_connection(new_inp_id, out_node_id)
        return alt

    def mod_func_add_node(self):
        alt = self._make_empty_alt()

        # select two nodes for the new one
        out_node_id = self.select_output(table='add_node_0')
        in_node_id = self.select_input(table='add_node_0', for_output=out_node_id, increment=True)
        # select activation
        activation = self.select_activation(f"{in_node_id}-{out_node_id}", increment=True)
        # create node object
        new_node = nodes.Linear(activation=activation)
        # add it to the alteration and connect
        new_node_id = alt.add_node_as_new(new_node)
        alt.add_connection(in_node_id, new_node_id)
        alt.add_connection(new_node_id, out_node_id)
        # TODO possibly break existing connection
        return alt

    def _mod_func_change_node(self):
        # change some properties of a node in the model
        # check that there are mutable nodes

        # plan
        # select node
        # change activation func
        # or
        # change bias
        # or
        # change some other properties
        pass


if __name__ == '__main__':
    from src.training.gym_agent import GymAgent
    import multiprocessing
    a = GymAgent('CartPole-v0')
    a.gr()
    data = a.create_worker_data()
    ss = multiprocessing.Value('i', 0)
    e = BruteForceWorker(a.worker_out_queue, ss, data)
    e.start()
