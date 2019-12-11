# gym agent class
import gym
import queue
import multiprocessing
from threading import Thread
from src.main.model import Model
from src.training.utils import test_graph_gym_m, init_graph, get_seeds, eval_meanstd_product
from src.training.workers.brute_force import BruteForceWorker as BFW
from time import sleep


class GymAgent:

    def __init__(self, env_id, model=None, weights=(-1.5, -0.5, 0.5, 1.5),
                 num_workers=multiprocessing.cpu_count(),
                 ):
        self.env_id = env_id
        self.weights = weights

        self.num_workers = num_workers
        self.__workers = []
        self.__worker_data = None
        self.__worker_timeout = 3  # seconds
        self.__worker_stop_signal = multiprocessing.Value('i', 0)
        self.worker_out_queue = multiprocessing.Queue()

        self.__training_stop_signal = True
        self.__training_thread = None

        # find io dimensions
        temp_env = gym.make(env_id)
        if isinstance(temp_env.action_space, gym.spaces.discrete.Discrete):
            self.num_outputs = temp_env.action_space.n
            self.__out_func = 'argmax'
        else:
            self.num_outputs = temp_env.action_space.shape[0]
            self.__out_func = 'none'

        self.num_inputs = temp_env.observation_space.shape[0]

        # init model
        if model is None:
            self.model = Model(self.num_inputs, self.num_outputs)
        else:
            self.model = model

        self.best_score = None
        self.worst_seed_id = None
        self.worst_seed_score = None

    def start_training(self):
        # wrap in thread
        self.__training_thread = Thread(target=self.__train)
        self.__training_thread.start()

    def __get_ready(self):
        # eval model in current useless state
        graph = self.model.make_graph()
        init, out = init_graph(graph=graph, out_func=self.__out_func)
        seeds = get_seeds(env_id=self.env_id, amount=5)
        scores = test_graph_gym_m(weights=self.weights, graph=graph, init=init, out=out,
                                  env_id=self.env_id, seeds=seeds)
        # now find out average
        self.best_score = eval_meanstd_product(list(scores.values()))
        # and worst case seed
        self.get_worst_seed(score_dict=scores, update_local=True)
        # create data to pass to workers
        self.create_worker_data(update_local=True)

    def gr(self):
        self.__get_ready()

    def __train(self):

        assert self.__training_stop_signal, "Attempt to launch training while it is already in progress."

        if self.best_score is None:
            self.__get_ready()

        self.recreate_queue()  # just in case previous exit was by emergency
        # begin training
        emergency_exit = False
        self.__training_stop_signal = False
        print('Training started.')
        while not self.__training_stop_signal:
            # launch workers
            self.launch_workers(self.__worker_data)
            # and wait for something to come out
            new_alteration = None
            while new_alteration is None:
                # check if stop signal was received
                if self.__training_stop_signal:
                    emergency_exit = True
                    break
                try:  # to get something out of the queue
                    new_alteration = self.worker_out_queue.get_nowait()
                except queue.Empty:
                    sleep(1)
                    continue
                else:
                    break
            if emergency_exit:
                break
            # print changes
            self.print_alt(new_alteration)
            # now shutdown workers
            st = Thread(target=self.stop_workers)
            st.start()
            # apply changes
            new_alteration.apply(target_model=self.model)
            # update scores
            self.get_worst_seed(score_dict=new_alteration.score_data, update_local=True)
            self.best_score = new_alteration.score_data['avg']
            # update worker data
            self.create_worker_data(update_local=True)
            # join thread
            st.join()
            # empty queue
            self.recreate_queue()

    def stop_training(self):
        # stop training loop
        self.__training_stop_signal = True
        self.stop_workers(silent=False)
        self.__training_thread.join()
        print('Training stopped.')

    def launch_workers(self, data):
        # TODO launch different workers in accordance with their performance
        # for now, just run a bunch of BFWs

        self.__workers = []  # shift old workers to garbage collector
        self.__worker_stop_signal.value = 0  #
        for n in range(self.num_workers):
            self.__workers.append(BFW(out_queue=self.worker_out_queue,
                                      exit_signal=self.__worker_stop_signal,
                                      data=data))
            self.__workers[-1].start()

    def stop_workers(self, silent=True):
        if not silent:
            print(f"Stopping {self.num_workers} workers...")
        self.__worker_stop_signal.value = 1  # this'll make workers to stop gracefully
        sleep(self.__worker_timeout)
        for w in self.__workers:
            if w.is_alive():
                if not silent:
                    print(f"{w.name} is still working. Waiting...")
                w.join()
            else:
                if not silent:
                    print(f"{w.name} stopped.")
                continue

    def create_worker_data(self, update_local=False):
        data = dict()
        data['model'] = self.model
        data['weights'] = self.weights
        data['out_func'] = self.__out_func
        data['env_id'] = self.env_id
        data['env_seed'] = self.worst_seed_id
        data['target_score'] = self.worst_seed_score
        data['best_score'] = self.best_score

        if update_local:
            self.__worker_data = data
        return data

    def get_worst_seed(self, score_dict, update_local=False):
        min_score = None
        seed = None

        for key, value in score_dict.items():
            if isinstance(key, str):
                # there could be some additional data, like l2_avg, l3_avg, etc. stored with key as string
                # seed data stored as int: [values]
                continue
            avg = eval_meanstd_product(value)
            if min_score is None or avg < min_score:
                min_score = avg
                seed = key
            else:
                continue

        if update_local:
            self.worst_seed_id = seed
            self.worst_seed_score = min_score

        return seed, min_score

    def recreate_queue(self):
        self.worker_out_queue = multiprocessing.Queue()

    def print_alt(self, alt):
        line_len = 20
        avg_score = alt.score_data['avg']
        worst_seed_score = alt.score_data['l2_avg']
        print("="*line_len)
        print(f"New alteration applied | AVG: {avg_score} | WSS: {worst_seed_score}")
        print("-"*line_len)
        print(f"Worker: {alt.wname}")
        print(f"PID: {alt.pid}")
        print("Changes:")
        for line in alt.changelog:
            print(line)
        print("-"*line_len)


if __name__ == "__main__":
    a = GymAgent("CartPole-v0", num_workers=1)


