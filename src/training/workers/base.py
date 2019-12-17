from multiprocessing import Process


class Worker(Process):

    def __init__(self, out_queue, exit_signal, data, error_queue=None):
        super(Worker, self).__init__(target=self.job)
        self.stop = exit_signal
        self.oq = out_queue
        self.data = data
        self.model = data['model']
        if error_queue is not None:
            self.error_queue = error_queue

