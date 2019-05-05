from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
from itertools import chain
from collections import defaultdict


class SarsaAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.episode_reward = 0
        self.episode_rewards = []
        self.total_steps = 0
        self.state = config.state_normalizer(self.task.reset())
        self.info = self.task.get_info()
        self.action = self.select_action(self.state, self.info)

    def close(self):
        pass

    def eval_step(self, state, info):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state, info)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def select_action(self, state, info):
        config = self.config
        q_values = self.network(state, info)
        q_values = to_np(q_values).flatten()
        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        return action 

    def step(self):
        config = self.config
        loss_dict = defaultdict(list)
        for _ in range(config.rollout_length):
            next_state, reward, done, next_info = self.task.step([self.action])
            next_state = config.state_normalizer(next_state)
            next_action = self.select_action(next_state, next_info)
            self.episode_reward += reward # for logging
            if done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
            q = self.network(self.state, self.info)[:1, self.action]
            if config.offline:
                next_q = (float(reward) + float(config.discount * (1 - done)) * self.network(next_state, next_info).max(1)[0]).detach()
            else:
                next_q = (float(reward) + float(config.discount * (1 - done)) * self.network(next_state, next_info)[:1, next_action]).detach()
            loss = F.mse_loss(q, next_q)
            loss_dict['MSE'].append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()           
            self.state = next_state
            self.info = next_info
            self.action = next_action

        self.total_steps += config.rollout_length
        for k, v in loss_dict.items():
            config.logger.add_scalar(tag=k, value=torch.mean(tensor(v)), step=self.total_steps)
