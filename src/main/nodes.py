import tensorflow as tf
from src.main.activations import activations


class Node(tf.Module):

    type = None
    type_id = None
    support_inputs = False
    support_outputs = False

    def __init__(self):
        super(Node, self).__init__(self.name)
        self.id = None
        self.inputs = []
        self.outputs = []

    @property
    def name(self):
        try:
            return f"{self.type}_{self.id}"
        except AttributeError:
            return f"{self.type}_NoID"

    def __str__(self):
        return self.name

    @property
    def num_inputs(self):
        return self.inputs.__len__()

    @property
    def specs(self):
        specs = {
            "id": self.id,
            "type": self.type,
            "type_id": self.type_id,
        }
        if self.inputs:
            specs['inputs'] = [inp.id for inp in self.inputs]
        if self.outputs:
            specs['outputs'] = [out.id for out in self.outputs]
        return specs

    @classmethod
    def from_specs(cls, specs):
        return cls()

    def drop_connections(self):
        if self.support_inputs:
            self.inputs = []
        if self.support_outputs:
            self.outputs = []

    def clone(self):
        return self.from_specs(self.specs)


class MainInput(Node):

    def __init__(self, num_inputs):
        super(MainInput, self).__init__()
        self.__num_inputs = num_inputs

    def to_graph(self, graph):
        with graph.as_default():
            placeholder = tf.keras.backend.placeholder(shape=(self.__num_inputs,), name='MainInput', dtype=tf.float32)
            return tf.identity_n(tf.unstack(placeholder), name="Inp")


class MainOutput(Node):

    def __init__(self, num_outputs):
        super(MainOutput, self).__init__()
        self.num_outputs = num_outputs

    def to_graph(self, graph):
        with graph.as_default():
            tensors = [graph.get_tensor_by_name(f"Out_{i}:0") for i in range(self.num_outputs)]
            return tf.stack(tensors, name="MainOutput")


class SharedWeight(Node):

    def __init__(self):
        super(SharedWeight, self).__init__()

    def to_graph(self, graph, max_len):
        with graph.as_default():
            placeholder = tf.keras.backend.placeholder(shape=(), dtype=tf.float32, name='sw')
            weights = []
            for i in range(int(max_len)+1):
                weights.append(tf.fill((i,), placeholder))
            return tf.identity_n(weights, "shared_weight")


class Input(Node):
    type = "Input"
    type_id = 0
    support_outputs = True

    def __init__(self, inp_idx):
        super(Input, self).__init__()
        self.inp_idx = inp_idx
        self.inputs = ()

    def to_graph(self, graph):
        with graph.as_default():
            return tf.identity(graph.get_tensor_by_name(f"Inp:{self.inp_idx}"), name=self.name)

    @property
    def num_inputs(self):
        return self.__num_inputs

    @property
    def specs(self):
        specs = super(Input, self).specs
        specs['inp_idx'] = self.inp_idx
        return specs

    @classmethod
    def from_specs(cls, specs):
        return cls(specs['inp_idx'])


class Output(Node):
    type = "Output"
    type_id = 1
    support_inputs = True

    def __init__(self, out_idx):
        super(Output, self).__init__()
        self.outputs = ()
        self.out_idx = out_idx

    def to_graph(self, graph, inputs):
        with graph.as_default():
            if inputs:
                return tf.identity(tf.add_n(inputs), name=f"Out_{self.out_idx}")
            else:
                return tf.constant(float('-inf'), shape=(), name=f"Out_{self.out_idx}")

    @property
    def specs(self):
        specs = super(Output, self).specs
        specs['out_idx'] = self.out_idx
        return specs

    @classmethod
    def from_specs(cls, specs):
        return cls(specs['out_idx'])


class Linear(Node):
    type = "Linear"
    type_id = 2
    support_inputs = True
    support_outputs = True

    def __init__(self, activation='relu', bias=0):
        super(Linear, self).__init__()
        assert activation in activations, f"There is no activation named {activation}"
        self.activation = activations[activation]
        self.activation_name = activation
        self.bias = bias

    def to_graph(self, graph, inputs, weights='shared'):
        self.name_scope.__init__(name=self.name)
        with graph.as_default(), self.name_scope:
            # set weights
            if weights == 'shared':
                # use shared weight
                w = graph.get_tensor_by_name(f"shared_weight:{len(inputs)}")
            elif weights is None or weights == 'none':
                # init random weights
                w = tf.Variable(tf.random.uniform((self.num_inputs,), -1., 1.),
                                trainable=True, name='w', dtype=tf.float32)
            else:
                # use passed weights
                w = tf.Variable(weights, trainable=True, name='w', dtype=tf.float32)

            b = tf.Variable(self.bias, trainable=True, name='b', dtype=tf.float32)
            out = self.activation(tf.tensordot(inputs, w, axes=1) + b)
        return out

    @property
    def specs(self):
        specs = super(Linear, self).specs
        specs['activation'] = self.activation_name
        specs['bias'] = self.bias
        return specs

    @classmethod
    def from_specs(cls, specs):
        act = specs['activation']
        bias = specs['bias']
        return cls(activation=act, bias=bias)


def create_from_specs(specs):
    node_type = specs['type_id']
    if node_type == 0:
        return Input.from_specs(specs)
    elif node_type == 1:
        return Output.from_specs(specs)
    elif node_type == 2:
        return Linear.from_specs(specs)
    else:
        raise AttributeError(f"Unknown node type: {node_type}")
