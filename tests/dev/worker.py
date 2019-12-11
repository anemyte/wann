from multiprocessing import Process, Value
from ctypes import c_bool, c_int
import time


class Main:

    def __init__(self, num_workers):
        self.data = Value(c_int, 1)
        self.ss = Value(c_bool, False)
        self.workers = [Worker(self.data, self.ss) for _ in range(num_workers)]

    def start_workers(self):
        for w in self.workers:
            w.start()

    def terminate_workers(self):
        for w in self.workers:
            w.terminate()


class Worker(Process):

    def __init__(self,  data, ss):
        super(Worker, self).__init__(target=self.job)
        self.data = data
        self.stop_signal = ss

    def job(self):
        while self.stop_signal.value:
            print(self.data.value)
            time.sleep(5)


if __name__ == "__main__":
    man = Main(2)

# ============================
# result:
# data is not shared unless using a specific class for shared memory
