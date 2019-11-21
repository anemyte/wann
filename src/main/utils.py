from pandas import DataFrame


class IOTable(DataFrame):

    def __init__(self, *args, **kwargs):
        super(IOTable, self).__init__(*args, **kwargs)
        self.__nodes = []

    @property
    def inputs(self):
        return self.columns.values

    @property
    def outputs(self):
        return self.index.values

    def add_node(self, node):
        if node.id in self.__nodes:
            return
        if node.support_inputs:
            self.loc[node.id] = 0
        if node.support_outputs:
            self[node.id] = 0
        if node.support_inputs and node.support_outputs:
            self.at[node.id, node.id] = None
        self.__nodes.append(node.id)

    def add_nodes(self, node_list):
        for node in node_list:
            self.add_node(node)

    def remove_node(self, node):
        if node.id not in self.__nodes:
            return
        if node.id in self.index:
            self.drop(index=node.id, inplace=True)
        if node.id in self.columns:
            self.drop(columns=node.id, inplace=True)
        self.__nodes.remove(node.id)

    def remove_nodes(self, node_list):
        for node in node_list:
            self.remove_node(node)

    def argmin_inputs(self, out_id):
        return self.columns[self.loc[out_id] == self.loc[out_id].min()].values

    def argmax_inputs(self, out_id):
        return self.columns[self.loc[out_id] == self.loc[out_id].max()].values

    def argmin_outputs(self, inp_id):
        return self.index[self[inp_id] == self[inp_id].min()].values

    def argmax_outputs(self, inp_id):
        return self.index[self[inp_id] == self[inp_id].max()].values
