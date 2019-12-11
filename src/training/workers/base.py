from threading import Thread


class Worker(Thread):

    def __init__(self, out_queue, exit_signal, data):
        super(Worker, self).__init__(target=self.job)
        self.stop = exit_signal
        self.oq = out_queue
        self.data = data
        self.model = data['model']
