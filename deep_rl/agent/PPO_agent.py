#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.infos = self.task.get_info() # store task_ids

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, keys=['info'])
        states = self.states
        infos = self.infos
        for t in range(config.rollout_length):
            prediction = self.network(states, infos)
            next_states, rewards, terminals, next_infos = self.task.step(to_np(prediction['a']))
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states),
                         'info': tensor_dict(infos)}) # cat?
            states = next_states
            infos = next_infos

        self.states = states
        self.infos = infos
        prediction = self.network(states, infos) # what is this for? Just fill in blank?
        storage.add(prediction)
        storage.placeholder() # fill in keys with no value

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        # advantages <- adv
        # cat data from all workers together (mix)
        states, actions, log_probs_old, returns, advantages, infos = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv', 'info'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                sampled_infos = {k: v[batch_indices] for k, v in infos.items()}

                prediction = self.network(sampled_states, sampled_infos, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                print('policy: {:.2f}, value: {:.2f}, network: {:2f}'.format(policy_loss, value_loss, self.network.loss()))
                self.opt.zero_grad()
                (policy_loss + value_loss + 0.05 * self.network.loss()).backward() # network loss collect loss in the middle
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
