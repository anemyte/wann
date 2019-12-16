from src.main import nodes
from collections import defaultdict, deque
import json
import tensorflow as tf
from multiprocessing import Pool, cpu_count
from src.main.utils import IOTable


class Model:

    def __init__(self, num_inputs, num_outputs, init_empty=False):

        self.__num_inputs = num_inputs
        self.__num_outputs = num_outputs

        self.__nodes = {}
        self.__nodes_by_type = defaultdict(set)
        self.__available_ids = []

        self.__input_connections = {}  # node.id - node.inputs relationship
        self.__output_connections = {}  # node.id - node.outputs relationship

        # Create mandatory nodes
        self.main_input = nodes.MainInput(num_inputs=num_inputs)
        self.shared_weight = nodes.SharedWeight()
        self.main_output = nodes.MainOutput(num_outputs=num_outputs)

        if not init_empty:
            # Create input nodes
            for i in range(num_inputs):
                self.add_node(nodes.Input(i))

            # Create output nodes
            for i in range(num_outputs):
                self.add_node(nodes.Output(i))

        self.best_score = None
        self.alterations = []

    # ================================================================
    # PUBLIC GET METHODS

    @property
    def num_inputs(self):
        return self.__num_inputs

    @property
    def num_outputs(self):
        return self.__num_outputs

    @property
    def nodes(self):
        return list(self.__nodes.values())

    @property
    def input_nodes(self):
        return list(self.__nodes_by_type[0])

    @property
    def output_nodes(self):
        return list(self.__nodes_by_type[1])

    @property
    def output_nodes_ids(self):
        return [node.id for node in self.__nodes_by_type[1]]

    @property
    def hidden_nodes(self):
        return list(self.__nodes_by_type[2])  # there will be more types

    @property
    def hidden_nodes_ids(self):
        return [node.id for node in self.hidden_nodes]

    @property
    def non_input_nodes(self):
        return self.output_nodes + self.hidden_nodes

    @property
    def non_output_nodes(self):
        return self.input_nodes + self.hidden_nodes

    @property
    def input_connections(self):
        inp_conn = {}
        for k, v in self.__input_connections.items():
            if v:
                inp_conn[k] = [n.id for n in v]
            else:
                inp_conn[k] = []
        return inp_conn

    @property
    def output_connections(self):
        out_conn = {}
        for k, v in self.__output_connections.items():
            if v:
                out_conn[k] = [n.id for n in v]
            else:
                out_conn[k] = []
        return out_conn

    def get_adjacency_matrix(self):
        matrix = IOTable()
        matrix.add_nodes(self.__nodes.values())

        for node in self.__nodes.values():
            if node.inputs:
                # add existing connections
                for inp in node.inputs:
                    matrix.at[node.id, inp.id] += 1
            if node.inputs and node.outputs:
                # mark impossible connections to avoid recursion
                possible_inputs = self.get_possible_input_ids(node, exclude_existing=False)
                # disable everything that's not in possible inputs
                matrix.loc[node.id, set(matrix.inputs).difference(possible_inputs)] = None

        return matrix

    def get_node_by_id(self, id_):
        return self.__nodes[id_]

    def get_id_for_new_node(self):
        if self.__available_ids:
            return self.__available_ids.pop()
        elif self.__nodes.__len__() not in self.__nodes:
            return self.__nodes.__len__()
        else:
            new_id = self.__nodes.__len__() + 1
            while new_id in self.__nodes:
                new_id += 1
            return new_id

    def get_possible_inputs(self, node, exclude_existing=True):
        # if node_id passed, switch it for actual node object
        if isinstance(node, int):
            node = self.__nodes[node]
        else:
            node = node

        if isinstance(node, nodes.Input):  # Input nodes do not support inputs, lol
            return []

        # create list of possible inputs
        lst = self.non_output_nodes
        # remove self from the list
        if node in lst:
            lst.remove(node)
        # remove all nodes that depend on this one
        dq = deque()
        dq.extend(node.outputs)
        while dq:
            out_node = dq.popleft()
            dq.extend(out_node.outputs)
            if out_node in lst:
                lst.remove(out_node)
        # remove currently used inputs as well
        if exclude_existing:
            for inp in node.inputs:
                lst.remove(inp)
        return lst

    def get_possible_input_ids(self, node, exclude_existing=True):
        possible_input_nodes = self.get_possible_inputs(node, exclude_existing)
        if possible_input_nodes:
            return [node.id for node in possible_input_nodes]
        else:
            return []

    # ================================================================
    # PUBLIC MODIFICATION METHODS

    def add_node(self, node):
        id_ = self.get_id_for_new_node()
        self._add_node_with_id(node, id_)
        return id_

    def remove_node(self, node):
        if isinstance(node, int):
            node = self.__nodes[node]

        # remove connections
        if node.outputs:
            for out in node.outputs:
                out.inputs.remove(node)
        if node.inputs:
            for inp in node.inputs:
                inp.outputs.remove(node)

        self.__nodes.pop(node.id)
        self.__nodes_by_type[node.type].remove(node)
        self.__nodes_by_type[node.type_id].remove(node)
        self.__input_connections.pop(node.id)
        self.__output_connections.pop(node.id)
        self.__available_ids.append(node.id)

    def connect_by_id(self, previous_node_id, next_node_id):
        previous_node = self.__nodes[previous_node_id]
        next_node = self.__nodes[next_node_id]
        next_node.inputs.append(previous_node)
        previous_node.outputs.append(next_node)

    def disconnect_by_id(self, previous_node_id, next_node_id):
        previous_node = self.__nodes[previous_node_id]
        next_node = self.__nodes[next_node_id]
        next_node.inputs.remove(previous_node)
        previous_node.outputs.remove(next_node)

    # ================================================================
    # PRIVATE MODIFICATION METHODS

    def _add_node_with_id(self, node, id_):
        node.id = id_
        self.__nodes[id_] = node
        self.__nodes_by_type[node.type].add(node)
        self.__nodes_by_type[node.type_id].add(node)
        self.__input_connections[node.id] = node.inputs
        self.__output_connections[node.id] = node.outputs

        if node.inputs:
            for inp in node.inputs:
                inp.outputs.append(node)
        if node.outputs:
            for out in node.outputs:
                out.inputs.append(node)

    def _restore_serialized_nodes(self, node_list):
        # create unconnected instances first
        for ns in node_list:
            node = nodes.create_from_specs(ns)
            self._add_node_with_id(node, ns['id'])

        # connect nodes
        for ns in node_list:
            node = self.__nodes[ns['id']]
            if 'inputs' in ns:
                for inp in ns['inputs']:
                    node.inputs.append(self.__nodes[inp])
                    self.__nodes[inp].outputs.append(node)

    def drop_all_connections(self):
        for node in self.__nodes.values():
            node.drop_connections()

    # ================================================================
    # PUBLIC EXPORT METHODS

    def to_json(self, file_path):
        data = self.to_dict()
        dump = json.dumps(data)
        with open(file_path, 'w') as file:
            file.write(dump)

    def to_dict(self):
        data = {
            "num_inputs": self.__num_inputs,
            "num_outputs": self.__num_outputs,
            "nodes": [node.specs for node in self.nodes],
            "best_score": self.best_score,
        }
        return data

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as file:
            data = json.loads(file.read())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, dict):
        new_instance = cls(dict['num_inputs'], dict['num_outputs'], init_empty=True)
        new_instance._restore_serialized_nodes(dict['nodes'])
        new_instance.best_score = dict['best_score']
        return new_instance

    def clone(self):
        data = self.to_dict()
        return self.from_dict(data)

    def clone_n(self, num):
        as_dict = self.to_dict()
        pool = Pool(cpu_count())
        out = []
        for _ in range(num):
            pool.apply_async(Model.from_dict, args=[as_dict], callback=out.append)
        pool.close()
        pool.join()
        return out

    def make_graph(self):
        if not tf.executing_eagerly:
            tf.compat.v1.disable_eager_execution()

        graph = tf.Graph()
        tensors = defaultdict(list)
        self.main_input.to_graph(graph)
        self.shared_weight.to_graph(graph, self.get_adjacency_matrix().sum(axis=1).max())

        # create nodes
        for node in self.__nodes.values():
            self.__add_node_to_graph(graph, tensors, node)

        self.main_output.to_graph(graph)
        return graph

    # ================================================================
    # PRIVATE EXPORT METHODS

    def __add_node_to_graph(self, graph, tensors, node):
        if node.id in tensors:
            return
        inputs = []
        for inp in node.inputs:
            if inp.id not in tensors:
                self.__add_node_to_graph(graph, tensors, inp)
                inputs.append(tensors[inp.id])
            else:
                inputs.append(tensors[inp.id])
        if node.type_id == 2:  # Linear node type
            node_out_tensors = node.to_graph(graph, inputs)  # + weights and bias later
        elif node.type_id == 1:  # Output node type
            node_out_tensors = node.to_graph(graph, inputs)
        elif node.type_id == 0:  # Input node type
            node_out_tensors = node.to_graph(graph)
        tensors[node.id] = node_out_tensors

    # ================================================================
    # DEPRECATED

    def get_maxlen(self, collection='inputs'):
        if collection.upper() == 'INPUTS':
            return len(max(self.__input_connections.values(), key=len))
        elif collection.upper() == 'OUTPUTS':
            return len(max(self.__output_connections.values(), key=len))
        else:
            raise NotImplemented(f"Unsupported collection: {collection}.")
