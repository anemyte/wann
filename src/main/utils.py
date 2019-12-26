from pandas import DataFrame
import numpy as np


class IOTable(DataFrame):

    def __init__(self, *args, **kwargs):
        super(IOTable, self).__init__(*args, **kwargs)

    @property
    def inputs(self):
        return self.columns.values

    @property
    def outputs(self):
        return self.index.values

    @property
    def nodes(self):
        return set(self.inputs).union(self.outputs)

    def add_node(self, node, as_input=True, as_output=True):
        if node.id in self.nodes:
            return
        if node.support_outputs and as_input:
            self[node.id] = 0
        if node.support_inputs and as_output:
            self.loc[node.id] = 0
        if node.support_inputs and node.support_outputs and as_input and as_output:
            self.at[node.id, node.id] = None

    def add_nodes(self, node_list, as_input=True, as_output=True):
        for node in node_list:
            self.add_node(node, as_input, as_output)

    def remove_node(self, node):
        if node.id in self.index:
            self.drop(index=node.id, inplace=True)
        if node.id in self.columns:
            self.drop(columns=node.id, inplace=True)

    def remove_nodes(self, node_list):
        for node in node_list:
            self.remove_node(node)

    def absmin(self):
        return self.min().min()

    def copy(self, deep=True):
        dt = super(IOTable, self).copy(deep=deep)
        new_instance = IOTable(dt)
        return new_instance

    def get_connections(self):
        return self[self > 0].stack().index.tolist()

    def get_possible_connections(self):
        return self[self == 0].stack().index.tolist()

    def get_input_connections_for(self, node_id):
        slc = self.loc[node_id]
        return slc[slc > 0].index.tolist()
