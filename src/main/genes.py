from collections import deque


class Gene:

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


class AddNode(Gene):

    def __init__(self, model, node):
        super(AddNode, self).__init__(model)
        self.new_node = node

    def __call__(self, target, apply=True):
        if apply:
            target.add_node(self.new_node)
        else:
            target.remove_node(self.new_node)


class AddConnection(Gene):

    def __init__(self, model, io_mappings):
        super(AddConnection, self).__init__(model)
        self.io_mappings = io_mappings

    def __call__(self, target, apply=True):
        for i, o in self.io_mappings:
            if apply:
                target.connect_by_id(i, o)
            else:
                target.disconnect_by_id(i, o)


