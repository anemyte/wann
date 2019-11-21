from src.main.model import Model


class WANN:

    def __init__(self, num_inputs, num_outputs):
        self.model = Model(num_inputs, num_outputs)
