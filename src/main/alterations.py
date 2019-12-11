from collections import deque
import numpy as np
import string
import random
import tensorflow as tf


class Alteration:

    def __init__(self, model):
        self.live_model = model
        self.test_model = model.clone()
        self.actions = deque()
        self.actions.append(self)
        self.scores = []  # placeholder for performance evaluation
        self.graph = None

    def apply(self):
        # shortcut for apply_to(live_model)
        self.apply_to(self.live_model)

    def revert(self):
        # shortcut for revert_for(live_model)
        self.revert_for(self.live_model)

    def apply_to(self, target):
        for action in self.actions:
            action(target)

    def revert_for(self, target):
        self.actions.reverse()
        for action in self.actions:
            action(target, apply=False)
        self.actions.reverse()

    def __call__(self, target, apply=True):
        # placeholder for actual action, done in sub-classes
        pass

    def create_graph(self):
        self.apply_to(self.test_model)
        self.graph = self.test_model.make_graph()
        self.revert_for(self.test_model)
        return self.graph

    def make_graph(self):
        return self.create_graph()


class AddNode(Alteration):

    def __init__(self, model, node):
        super(AddNode, self).__init__(model)
        self.new_node = node

    def __call__(self, target, apply=True):
        if apply:
            target.add_node(self.new_node)
        else:
            target.remove_node(self.new_node)


class AddConnection(Alteration):

    def __init__(self, model, inp_node, out_node):
        super(AddConnection, self).__init__(model)
        self.input_node = inp_node
        self.output_node = out_node

    def __call__(self, target, apply=True):
        if apply:
            target.connect_by_id(self.input_node.id, self.output_node.id)
        else:
            target.disconnect_by_id(self.input_node.id, self.output_node.id)


class AlterationV2:

    # purpose
    # - to hold a set of changes over associated model
    # - to build graph with the changes but without changing the model
    # - to apply the changes to the model

    def __init__(self, model):

        # purpose
        # - create instance and containers for actions
        # - save reference to the model

        self.graph = None
        self.model = model
        self.new_nodes = dict()  # name - node pairs
        self.repl_nodes = dict()
        self.node_aliases = dict()  # old_node_id -> new_node_id
        self.io = model.get_adjacency_matrix()  # io table
        self.is_applied = False  # to protect from several consecutive calls
        self.score_data = None
        self.changelog = []

    def make_graph(self):
        # purpose
        # - create tensorflow Graph object
        # - use data in stored actions to build graph that differs from the model

        # some ideas how to make it
        # - i can make an adjacency table from model to find out how the nodes are connected
        # - then i can select two unconnected nodes and get their ids
        # - then i can take note that this pair was tested (and make that kind of information to flow between launches)
        # - then i can pass io mapping to this method
        # - this method init main input and sw
        # - then it goes over columns in io mapping
        # - if it stuck into node that weren't added, the method creates that node before going further
        # - while creating nodes the method saves their outputs as tensors
        # - while adding nodes the method first looks into nodes in this object's storage (in case of modified nodes)
        # the produced graph is saved as self.graph and returned back

        graph = tf.Graph()
        tensors = dict()
        self.model.main_input.to_graph(graph)
        self.model.shared_weight.to_graph(graph, len(self.io.columns))
        for node_id in self.io.outputs:
            self.__add_to_graph(node_id, tensors, graph)

        self.model.main_output.to_graph(graph)
        return graph

    def __add_to_graph(self, id, tensors, graph):
        if id in tensors:  # that is if the node has been added already
            return

        if id in self.new_nodes:
            node = self.new_nodes[id]
        elif id in self.repl_nodes:
            node = self.repl_nodes[id]
        else:
            node = self.model.get_node_by_id(id)

        # if node does not support input there is nothing more to do
        if not node.support_inputs:
            tensor = node.to_graph(graph)
            tensors[id] = tensor
            return tensor

        node_inputs = []
        input_ids = self.io.loc[id] == 1
        for idx, val in zip(input_ids.index, input_ids.values):
            if val:  # that evaluates to true if this node receives input from another
                if idx not in tensors:  # if the other node wasn't added to the graph
                    node_inputs.append(self.__add_to_graph(idx, tensors, graph))
                else:
                    node_inputs.append(tensors[idx])
        tensor = node.to_graph(graph, node_inputs)
        tensors[id] = tensor
        return tensor

    def apply(self, target_model=None):
        # purpose
        # - save whatever changes this alteration has into the original model

        # for example
        # for each node in local container replace node in the model with a copy of the node if there is a node with the same id
        # for each new node in local container
        # 1. add a copy of it to the model and take reference of its new id
        # 2. save alias in this object that old id refers to the new id
        # drop all existing connections in the model (to avoid possible bugs with bad links)
        # then for each index in the local io table
        # - connect to node where 1

        if self.is_applied:
            return
        else:
            self.is_applied = True

        if target_model is None:
            target_model = self.model

        target_model.drop_all_connections()

        # replace nodes first
        for node in self.repl_nodes.values():
            # Do replace process
            # just replace an existing with a clone
            target_model._add_node_with_id(node.clone(), node.id)

        # add new nodes
        for node in self.new_nodes.values():
            # do addition process
            # 1. add clone to the model
            # 2. get new id
            # 3. register alias
            new_id = target_model.add_node(node.clone())
            self.node_aliases[node.id] = new_id

        # connect by local schema
        connections = self.io.get_connections()
        # for out_id, inp_id in connections
        for o, i in connections:
            # replace id with alias if present
            if i in self.node_aliases:
                i = self.node_aliases[i]
            if o in self.node_aliases:
                o = self.node_aliases[o]
            # do connect
            target_model.connect_by_id(i, o)

        target_model.alterations.append(self)

    def add_node_as_new(self, node):
        # create id for new node
        # save it in local storage
        tmp_id = self._create_temp_node_id()
        self.new_nodes[tmp_id] = node
        node.id = tmp_id
        self.io.add_node(node)

        logstr = f"Added new {node.__class__.__name__} node"
        try:
            logstr += f" with {node.activation_name}."
        except AttributeError:
            logstr += "."
        self.changelog.append(logstr)
        return tmp_id

    def add_node_as_replacement(self, node, id):
        # add node without creating id
        self.repl_nodes[id] = node

        # TODO maybe add more information for this change
        self.changelog.append(f"Replaced {self.model.get_node_by_id(id).__class__.__name__} with ID {id}.")

    def add_connection(self, prev, next):
        # adjust io table
        if np.isnan(self.io.at[next, prev]) or next not in self.io.outputs:
            raise AttributeError(f"{prev} cannot be connected to {next}")
        else:
            self.io.at[next, prev] = 1

        self.changelog.append(f"New connection: {prev}->{next}.")

    def remove_connection(self, prev, next):
        # adjust io table
        if np.isnan(self.io.at[next, prev]):
            raise AttributeError(f"{prev} cannot be connected to {next}")
        else:
            self.io.at[next, prev] = 0

        self.changelog.append(f"Removed connection between {prev} and {next}")

    def revert(self):
        # not necessary but could be handy
        # revert changes made to the model
        pass

    def _create_temp_node_id(self):
        new_id = max(self.io.nodes) + 1 * 100
        return new_id
