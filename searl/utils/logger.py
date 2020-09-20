import numpy as np
import time

from .base_handler import Handler
from .txt import LogTXT


class Logger(Handler):

    def __init__(self, log_dir):
        super().__init__()

        self.txt_logger = LogTXT(log_dir)

        self.timer = {}

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __call__(self, key, value=None, time_step=None, print_log=True):
        self.log(key, value, time_step, print_log)

    def log(self, key, value=None, time_step=None, print_log=True):
        if value is None:
            if print_log:
                if time_step is None:
                    self.txt_logger.log(key)
                else:
                    self.txt_logger.log(f"{key}-step:{time_step}")

        else:
            if time_step is None:
                self.txt_logger.log(f"{key}: {value}")
            else:
                self.txt_logger.log(f"{key}: {value}  step:{time_step}")

    def print_config(self, config, name="main"):
        if name == "main":
            self.log("#" * 20 + " CONFIG:")
        else:
            self.log(f"sub config {name:8}", np.unique([f"{attr} : {str(value)}  " for attr, value in config.get_dict.items()]).tolist())

        if hasattr(config, "sub_config"):
            for cfg in config.sub_config:
                self.print_config(getattr(config, cfg), cfg)

    def start_timer(self, name):
        self.log(f"##### {name}")
        self.timer[name] = time.time()

    def log_time(self, name):
        self.log(f"timer {name:8}", f"{time.time() - self.timer[name]:3.1f}s")

    def log_func(self, function, *args, **kwargs):
        self.start_timer(function.__name__)
        rslt = function(*args, **kwargs)
        self.log_time(function.__name__)
        return rslt

    def population_info(self, population_mean_fitness, population_var_fitness, population, num_frames, epoch):

        self.log("#### POPULATION INFO", epoch, time_step=num_frames)
        self.log('Population fitness', [ind.fitness[-1] for ind in population], time_step=num_frames)
        self.log('Population improve', [ind.improvement for ind in population], time_step=num_frames)
        self.log('Population var fit', [float(var) for var in population_var_fitness], time_step=num_frames)
        self.log('Actors hidden size', [[int(s) for s in ind.actor_config['hidden_size']] for ind in population], time_step=num_frames)
        self.log('Critic hidden size', [[int(s) for s in ind.critic_config['hidden_size']] for ind in population], time_step=num_frames)
        self.log('Mutation: ', [ind.train_log["mutation"] for ind in population], time_step=num_frames)

        best_idx = np.argmax(population_mean_fitness)
        self.log('mean_fitness', np.mean(population_mean_fitness), time_step=num_frames)
        self.log('best_fitness', population[best_idx].fitness[-1], time_step=num_frames)
        self.log('best_improve', population[best_idx].improvement, time_step=num_frames)
        self.log('best rl config', population[best_idx].rl_config.__str__(), time_step=num_frames)
        self.log('Best Actors hidden size', [int(s) for s in population[best_idx].actor_config['hidden_size']], time_step=num_frames)
        self.log('Best Critic hidden size', [int(s) for s in population[best_idx].critic_config['hidden_size']], time_step=num_frames)
