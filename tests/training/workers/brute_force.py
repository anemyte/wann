import unittest
import multiprocessing
import gym
from src.training import BruteForceWorker
from src import Model


class BFWorker(unittest.TestCase):

    def test_select_output_1(self):
        """
        Very basic i/o operations.
        """

        w = make_worker()
        test_table_name = 't'

        # case 0
        # no connections in the model, should return integer id of one of the output nodes
        possible_outputs = list(map(lambda x: x.id, w.model.non_input_nodes))
        case_0 = w.select_output(test_table_name, increment=True, for_input=0)
        self.assertIn(case_0, possible_outputs,
                      f"Selected output is not in the list of possible output nodes:\n"
                      f"Result: {case_0} should be in {possible_outputs}\n"
                      f"Table: {w.io_tables[test_table_name]}")

        # case 1
        # incrementation was successful
        self.assertIn(1, w.io_tables[test_table_name].values,
                      f"Incrementation failed:\n"
                      f"Result: {case_0}\n"
                      f"Table: {w.io_tables[test_table_name]}")

        # case 2
        # greedy must return something other than case_0
        case_2 = w.select_output(test_table_name, greedy=True)
        self.assertNotEqual(case_0, case_2,
                            f"Failed to obtain greedy result:\n"
                            f"Result: {case_0} should differ from {case_2}\n"
                            f"Table: {w.io_tables[test_table_name]}")

    def test_select_io_recursion(self):
        """
        Select only compatible nodes and avoid recursion.
        """
        w = make_worker()
        test_table_name = 't'
        # add some nodes
        num_new_nodes = 30
        for _ in range(num_new_nodes):
            w.mm_add_node().apply()
            w.io_tables['add_node_0'].add_node(w.model.nodes[-1])
        # test create graph, if nodes were added incorrectly, this may produce an infinite recursion
        try:
            graph = w.model.make_graph()
        except RecursionError:
            print(w.model.get_adjacency_matrix())




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


def make_worker_data_dict():
    """
    Create a dictionary of parameters to use in testing.
    """
    data = dict()
    data['env_id'] = "CartPole-v0"
    data['model'] = make_model_for_env(data['env_id'])
    data['weights'] = (1.5, 0.5, -0.5, -1.5)
    data['out_func'] = 'argmax'
    data['env_seed'] = 0
    data['target_score'] = 0.0
    data['best_score'] = 0.0
    return data


def make_worker():
    dd = make_worker_data_dict()
    ss = multiprocessing.Value('i', 0)
    oq = multiprocessing.Queue()
    worker = BruteForceWorker(out_queue=oq, exit_signal=ss, data=dd)
    return worker


if __name__ == '__main__':
    unittest.main()
