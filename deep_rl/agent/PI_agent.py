from ..network import *
from ..component import *
from .BaseAgent import *

from collections import defaultdict


# can only use PosAbstractor and TabularActor
class PIAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.total_steps = 0
        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.infos = self.task.get_info() # store task_ids
        self.k = 1 # update eps

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length, keys=['abs_s'])
        states = self.states
        infos = self.infos
        for t in range(config.rollout_length):
            prediction = self.network(states, infos)
            actions = to_np(prediction['a'])
            abs_s = self.network.abs_encoder.get_indices(states, infos)
            next_states, rewards, terminals, next_infos = self.task.step(actions)
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards),
                         'm': tensor(1 - terminals),
                         's': tensor(states),
                         'ns': tensor(next_states),
                         'info': infos,
                         'abs_s': abs_s,
                         })
            states = next_states
            infos = next_infos

        self.states = states
        self.infos = infos
        prediction = self.network(states, infos) # what is this for? Just fill in blank?
        storage.add(prediction)
        storage.placeholder() # fill in keys with no value

        q = tensor(np.zeros((config.eval_env.n_tasks, config.n_abs, config.action_dim)))
        count = tensor(np.zeros((config.eval_env.n_tasks, config.n_abs, config.action_dim)))
        returns = tensor(np.zeros(config.num_workers))
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            for j in range(config.num_workers):
                q[storage.info[i]['task_id'][j]][storage.abs_s[i][j]][storage.a[i][j]] += returns[j]
                count[storage.info[i]['task_id'][j]][storage.abs_s[i][j]][storage.a[i][j]] += 1
        q /= count
        new_policy = one_hot.encode(q.argmax(dim=2), config.action_dim)
        self.network.actor.policy += 1 / self.k * (new_policy - self.network.actor.policy)
        #advantages = tensor(np.zeros((config.num_workers, 1)))
        #returns = prediction['v'].detach()
        #for i in reversed(range(config.rollout_length)):
        #    returns = storage.r[i] + config.discount * storage.m[i] * returns
        #    if not config.use_gae:
        #        advantages = returns - storage.v[i].detach()
        #    else:
        #        td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
        #        advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
        #    storage.adv[i] = advantages.detach()
        #    storage.ret[i] = returns.detach()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.network.step() # do all adaptive update
        self.k += 1
