import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLTraining():

    def __init__(self, config, replay_sample_queue):
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.searl.seed)
        self.replay_sample_queue = replay_sample_queue

        self.args = config.td3

    @staticmethod
    def update_parameters(indi, replay_sample_queue, iterations):

        args = indi.rl_config

        gamma = args.gamma
        tau = args.tau

        actor = indi.actor
        actor_target = type(actor)(**actor.init_dict)
        actor_target.load_state_dict(actor.state_dict())
        actor.to(device)
        actor_target.to(device)
        actor.train()
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)

        critic_1 = indi.critic_1
        critic_1_target = type(critic_1)(**critic_1.init_dict)
        critic_1_target.load_state_dict(critic_1.state_dict())
        critic_1.to(device)
        critic_1_target.to(device)
        critic_1.train()
        critic_1_optim = torch.optim.Adam(critic_1.parameters(), lr=args.lr_critic_1)

        critic_2 = indi.critic_2
        critic_2_target = type(critic_2)(**critic_2.init_dict)
        critic_2_target.load_state_dict(critic_2.state_dict())
        critic_2.to(device)
        critic_2_target.to(device)
        critic_2.train()
        critic_2_optim = torch.optim.Adam(critic_2.parameters(), lr=args.lr_critic_2)

        for it in range(iterations):

            transistion_list = replay_sample_queue.get()

            state_batch = torch.stack([torch.Tensor(transition.state) for transition in transistion_list], dim=0)
            action_batch = torch.stack([torch.Tensor(transition.action) for transition in transistion_list], dim=0)
            next_state_batch = torch.stack([torch.Tensor(transition.next_state) for transition in transistion_list], dim=0)
            reward_batch = torch.stack([torch.Tensor(transition.reward) for transition in transistion_list], dim=0)
            done_batch = torch.stack([torch.Tensor(transition.done) for transition in transistion_list], dim=0)

            state = state_batch.to(device)
            action = action_batch.to(device)
            reward = reward_batch.to(device)
            done = 1 - done_batch.to(device)
            next_state = next_state_batch.to(device)

            with torch.no_grad():

                noise = (torch.randn_like(action) * args.td3_policy_noise).clamp(-args.td3_noise_clip, args.td3_noise_clip)

                next_action = (actor_target(next_state) + noise).clamp(-1, 1)

                target_Q1 = critic_1_target(torch.cat([next_state, next_action], 1))
                target_Q2 = critic_2_target(torch.cat([next_state, next_action], 1))
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * gamma * target_Q)

            current_Q1 = critic_1(torch.cat([state, action], 1))
            current_Q2 = critic_2(torch.cat([state, action], 1))

            critic_loss_1 = F.mse_loss(current_Q1, target_Q)
            critic_1_optim.zero_grad()
            critic_loss_1.backward()
            for p in critic_1.parameters():
                p.grad.data.clamp_(max=args.clip_grad_norm)
            critic_1_optim.step()

            critic_loss_2 = F.mse_loss(current_Q2, target_Q)
            critic_2_optim.zero_grad()
            critic_loss_2.backward()
            for p in critic_2.parameters():
                p.grad.data.clamp_(max=args.clip_grad_norm)
            critic_2_optim.step()

            if it % args.td3_update_freq == 0:

                actor_loss = -critic_1(torch.cat([state, actor(state)], 1))
                actor_loss = torch.mean(actor_loss)

                actor_optim.zero_grad()
                actor_loss.backward()
                for p in actor.parameters():
                    p.grad.data.clamp_(max=args.clip_grad_norm)
                actor_optim.step()

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(critic_1.parameters(), critic_1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(critic_2.parameters(), critic_2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        actor_optim.zero_grad()
        critic_1_optim.zero_grad()
        critic_2_optim.zero_grad()

        indi.actor = actor.cpu()
        indi.critic_1 = critic_1.cpu()
        indi.critic_2 = critic_2.cpu()

        indi.train_log['train_iterations'] = iterations
        indi.train_log.update(args.get_dict)

        return indi

    def train(self, population, eval_frames, pool):

        pop_id_lookup = [ind.index for ind in population]

        iterations = int(self.cfg.searl.train_frames_fraction * eval_frames)

        if self.cfg.searl.ind_memory:
            args_list = [(indi, indi.replay_memory, iterations) for indi in population]
        else:
            args_list = [(indi, self.replay_sample_queue, iterations) for indi in population]

        result_dicts = [pool.apply_async(self.update_parameters, args) for args in args_list]
        trained_pop = [res.get() for res in result_dicts]

        trained_pop = sorted(trained_pop, key=lambda i: pop_id_lookup.index(i.index))

        return trained_pop
