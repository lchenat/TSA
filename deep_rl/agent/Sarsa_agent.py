from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
from itertools import chain
from collections import defaultdict

def softmax(a):
    a = np.exp(a - a.max())
    return a / a.sum(-1)

class SarsaActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
            self._info = self._task.get_info()
        config = self.config
        q_values = self._network(config.state_normalizer(self._state), self._info)
        q_values = to_np(q_values).flatten()
        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), self._info, info]
        self._total_steps += 1
        self._state = next_state
        self._info = info
        return entry

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
        self.state = self.task.reset()
        self.info = self.task.get_info()
        self.action = self.select_action()

    def close(self):
        pass

    def eval_step(self, state, info):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state, info)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def select_action(self):
        config = self.config
        q_values = self.network(config.state_normalizer(self.state), self.info)
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
            next_action = self.select_action()
            self.episode_reward += reward # for logging
            if done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
            q = self.network(self.state, self.info)[:1, self.action]
            next_q = self.network(next_state, next_info)[:1, next_action]
            loss = F.mse_loss(q - config.discount * next_q, tensor(reward))
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
