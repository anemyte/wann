import unittest
from src.main.model import Model
from src.main.nodes import Linear
import random
import tensorflow as tf


class Basic(unittest.TestCase):

    num_inputs = 2
    num_outputs = 2
    num_hiddens = 0
    model = Model(num_inputs, num_outputs)

    @property
    def num_nodes(self):
        return self.num_inputs + self.num_outputs + self.num_hiddens

    def test_get_nodes(self, model=model):
        self.assertEqual(list, type(model.nodes), "Expected model to return a list of nodes")
        self.assertEqual(self.num_nodes, model.nodes.__len__(), f"Expected  {self.num_nodes} nodes")
        self.assertEqual(self.num_inputs, model.input_nodes.__len__(), f"Expected {self.num_inputs} input nodes")
        self.assertEqual(self.num_outputs, model.output_nodes.__len__(), f"Expected {self.num_outputs} output nodes")
        self.assertEqual(self.num_hiddens, model.hidden_nodes.__len__(), f"Expected {self.num_hiddens} hidden nodes")
        self.assertEqual(self.num_hiddens+self.num_outputs, model.non_input_nodes.__len__(),
                         f"Expected {self.num_hiddens+self.num_outputs} non-input nodes")
        self.assertEqual(self.num_hiddens+self.num_inputs, model.non_output_nodes.__len__(),
                         f"Expected {self.num_hiddens+self.num_inputs} non-output nodes")

    def test_add_nodes(self):
        new_node = Linear('relu')
        new_node.inputs.append(self.model.input_nodes[0])
        new_node.outputs.append(self.model.output_nodes[0])
        self.model.add_node(new_node)
        self.num_hiddens += 1
        self.assertIn(new_node, self.model.nodes, "New node not found in __nodes")
        self.assertIn(new_node, self.model.hidden_nodes, "New node not found in hidden nodes")
        self.test_get_nodes(model=self.model)
        self.model.remove_node(new_node)
        self.num_hiddens -= 1
        self.test_get_nodes(model=self.model)


class Export(unittest.TestCase):

    @staticmethod
    def create_sample_model(num_inputs=10, num_outputs=10, num_hidden=10):
        m = Model(num_inputs, num_outputs)
        for i in range(num_hidden):
            node = Linear('relu')
            node.inputs.append(random.choice(m.input_nodes))
            node.outputs.append(random.choice(m.output_nodes))
            m.add_node(node)
        return m

    @staticmethod
    def calc(graph):
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)
            result = sess.run("MainOutput:0", feed_dict={
                "MainInput:0": list(range(10)),
                "sw:0": 2,
            })
        return result

    def test_clone(self):
        original = self.create_sample_model()
        clone = original.clone()
        self.assertEqual(original.to_dict(), clone.to_dict(), "Dicts are not equal.")
        self.assertEqual(original.input_connections, clone.input_connections, "Inputs are not equal")
        self.assertEqual(original.output_connections, clone.output_connections, "Outputs are not equal")

    def test_save_load(self):
        original = self.create_sample_model()
        original.to_json('temp.json')
        clone = Model.from_json('temp.json')
        self.assertEqual(original.to_dict(), clone.to_dict(), "Dicts are not equal.")
        self.assertEqual(original.input_connections, clone.input_connections, "Inputs are not equal")
        self.assertEqual(original.output_connections, clone.output_connections, "Outputs are not equal")

    def test_calculation(self):
        original = self.create_sample_model()
        clone = original.clone()

        result_1 = self.calc(original.make_graph())
        result_2 = self.calc(clone.make_graph())
        self.assertListEqual(list(result_1), list(result_2), "Execution results are not equal.")


if __name__ == '__main__':
    unittest.main()

