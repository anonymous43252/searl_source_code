import os
import time
from pathlib import Path
import copy
import numpy as np
import torch
import torch.multiprocessing as mp
import gym

from searl.utils.supporter import Supporter
from searl.components.replay_memory import MPReplayMemory, ReplayMemory
from searl.components.evaluation import MPEvaluation
from searl.components.tournament_selection import TournamentSelection
from searl.components.mutation import Mutations
from searl.components.training import RLTraining
from searl.components.individual import Individual


class SEARL():

    def __init__(self, config, logger, checkpoint):

        self.cfg = config
        self.log = logger
        self.ckp = checkpoint

        torch.manual_seed(self.cfg.searl.seed)
        np.random.seed(self.cfg.searl.seed)

        self.log.print_config(self.cfg)

        if self.cfg.searl.ind_memory:
            push_queue = None
            sample_queue = None
        else:
            self.replay_memory = MPReplayMemory(capacity=self.cfg.searl.replay_memory_size, batch_size=self.cfg.td3.batch_size)
            push_queue = self.replay_memory.get_push_queue()
            sample_queue = self.replay_memory.get_sample_queue()
            self.log.log("initialize replay memory")

        self.eval = MPEvaluation(config=self.cfg, logger=self.log, push_queue=push_queue)

        self.tournament = TournamentSelection(config=self.cfg)

        self.mutation = Mutations(config=self.cfg)

        self.training = RLTraining(config=self.cfg, replay_sample_queue=sample_queue)

    def initial_population(self):
        self.log.log("initialize population")
        population = []
        for idx in range(self.cfg.searl.population_size):

            if self.cfg.searl.ind_memory:
                replay_memory = ReplayMemory(capacity=self.cfg.searl.replay_memory_size, batch_size=self.cfg.td3.batch_size)
            else:
                replay_memory = False

            if self.cfg.searl.init_random:

                min_lr = 0.00001
                max_lr = 0.005
                max_layer = 2
                min_nodes = 128
                max_nodes = 384

                actor_config = copy.deepcopy(self.cfg.actor.get_dict)
                critic_config = copy.deepcopy(self.cfg.critic.get_dict)
                rl_config = copy.deepcopy(self.cfg.td3)

                actor_config["activation"] = np.random.choice(['relu', 'tanh', 'elu'], 1)[0]
                critic_config["activation"] = np.random.choice(['relu', 'tanh', 'elu'], 1)[0]

                hidden_size_len = np.random.randint(1, max_layer + 1)
                actor_config["hidden_size"] = [np.random.randint(min_nodes, max_nodes + 1) for _ in range(hidden_size_len)]
                critic_config["hidden_size"] = actor_config["hidden_size"]

                lr_actor = np.exp(np.random.uniform(np.log(min_lr), np.log(max_lr), 1))[0]
                lr_critic = np.exp(np.random.uniform(np.log(min_lr), np.log(max_lr), 1))[0]

                rl_config.set_attr("lr_actor", lr_actor)
                rl_config.set_attr("lr_critic", lr_critic)
                self.log(f"init {idx} rl_config: ", rl_config.get_dict)
                self.log(f"init {idx} actor_config: ", actor_config)

            else:
                actor_config = copy.deepcopy(self.cfg.actor.get_dict)
                critic_config = copy.deepcopy(self.cfg.critic.get_dict)
                rl_config = copy.deepcopy(self.cfg.td3)

            indi = Individual(state_dim=self.cfg.state_dim, action_dim=self.cfg.action_dim,
                              actor_config=actor_config,
                              critic_config=critic_config,
                              rl_config=rl_config, index=idx, replay_memory=replay_memory)
            population.append(indi)
        return population

    def evolve_population(self, population, epoch=1, num_frames=0):

        frames_since_mut = 0
        num_frames = num_frames
        epoch = epoch

        ctx = mp.get_context('spawn')

        while True:
            pool = ctx.Pool(processes=self.cfg.searl.worker, maxtasksperchild=1000)
            epoch_time = time.time()
            self.log(f"##### START EPOCH {epoch}", time_step=num_frames)

            for ind in population:
                ind.train_log['epoch'] = epoch

            population_mean_fitness, population_var_fitness, eval_frames = \
                self.log.log_func(self.eval.evaluate_population, population=population, exploration_noise=self.cfg.td3.exploration_noise,
                                  total_frames=num_frames, pool=pool)
            num_frames += eval_frames
            frames_since_mut += eval_frames

            self.log.population_info(population_mean_fitness, population_var_fitness, population, num_frames, epoch)

            self.ckp.save_object(population, name="population")
            self.log.log("save population")
            if not self.cfg.searl.ind_memory:
                rm_dict = self.replay_memory.save()
                if isinstance(rm_dict, str):
                    self.log("save replay memory failed")
                else:
                    self.log("replay memory size", len(rm_dict['memory']))
                self.ckp.save_object([rm_dict], name="replay_memory")
                self.log("save replay memory")

            if num_frames >= self.cfg.searl.num_frames:
                break

            elite, population = self.log.log_func(self.tournament.select, population)
            test_fitness = self.eval.test_individual(elite, epoch)
            self.log(f"##### Best {epoch}", time_step=num_frames)
            self.log("best_test_fitness", test_fitness, num_frames)

            population = self.log.log_func(self.mutation.mutation, population)

            population = self.log.log_func(self.training.train, population=population, eval_frames=eval_frames, pool=pool)

            self.log(f"##### END EPOCH {epoch} - runtime {time.time() - epoch_time:6.1f}", time_step=num_frames)
            self.log("epoch", epoch, time_step=num_frames)
            self.log(f"##### ################################################# #####")
            self.cfg.expt.set_attr("epoch", epoch)
            self.cfg.expt.set_attr("num_frames", num_frames)
            epoch += 1

            pool.terminate()
            pool.join()

        self.log("FINISH", time_step=num_frames)
        self.replay_memory.close()

    def close(self):
        self.replay_memory.close()


def start_actor_critic_searl(config_dir, expt_dir):
    sup = Supporter(experiments_dir=expt_dir, config_dir=config_dir, count_expt=True)
    cfg = sup.get_config()
    log = sup.get_logger()

    env = gym.make(cfg.env.name)
    cfg.set_attr("action_dim", env.action_space.shape[0])
    cfg.set_attr("state_dim", env.observation_space.shape[0])

    searl = SEARL(config=cfg, logger=log, checkpoint=sup.ckp)

    population = searl.initial_population()
    searl.evolve_population(population)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='define setup')

    parser.add_argument('--expt_dir', type=str, default=False, help='Dir to store the experiment')
    parser.add_argument('--config_dir', type=str, default=False, help='Dir to find the config')
    args = parser.parse_args()

    os.environ["LD_LIBRARY_PATH"] = f"$LD_LIBRARY_PATH:{str(Path.home())}/.mujoco/mujoco200/bin:/usr/lib/nvidia-384"

    start_actor_critic_searl(config_dir=args.config_dir, expt_dir=args.expt_dir)
