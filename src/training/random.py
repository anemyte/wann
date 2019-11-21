import random
from src.main import nodes
from src.main.activations import activations
from src.main.alterations import AddNode


class RandomSearch:

    def __init__(self, wann, pop_size=10, gen_size=10):
        self.wann = wann
        self.m = self.wann.model
        self.pop_size = pop_size
        self.gen_size = gen_size
        self.methods = []

    def make_new_species(self, num=0):
        pass

    def new_node(self):
        # select node that will receive output from new one
        out_node = random.choice(self.m.output_nodes)
        while out_node.inputs and random.uniform(0, 1) > 0.5:
            possible_outputs = []
            for node in out_node.inputs:
                if isinstance(node, nodes.Input):
                    continue
                else:
                    possible_outputs.append(node)
            if possible_outputs:
                out_node = random.choice(possible_outputs)
            else:
                break
        possible_inputs = self.m.get_possible_inputs(out_node, exclude_existing=False)
        prev_node = random.choice(possible_inputs)
        new_node = nodes.Linear(activation=random.choice(list(activations.keys())))
        new_node.inputs.append(prev_node)
        new_node.outputs.append(out_node)
        alt = AddNode(self.wann.model, new_node)
        return alt
