from ..network import *
from ..component import *
from .BaseAgent import *

class ImitationAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network)
        self.total_steps = 0
        self.states = config.state_normalizer(self.task.reset()) # change
        self.infos = self.task.get_info() # store task_ids

        self.episode_rewards = []
        self.online_rewards = np.zeros(config.num_workers)

    def eval_step(self, state, info):
        return self.network(state, info)['a'][0]

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        info = env.get_info()
        total_rewards = 0
        while True:
            action = self.eval_step(state, info)
            state, reward, done, _ = env.step([action])
            total_rewards += reward[0]
            if done[0]:
                break
        return total_rewards

    def eval_episodes(self):
        rewards = []
        for ep in range(self.config.eval_episodes):
            rewards.append(self.eval_episode())
        self.config.logger.info('evaluation episode return: %f(%f)' % (
            np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))))
        self.config.logger.add_scalar(tag='eval_return', value=np.mean(rewards), step=self.total_steps)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        infos = self.infos
        for _ in range(config.rollout_length):
            prediction = self.network(states, infos)
            #next_states, rewards, terminals, _ = self.task.step(to_np(prediction['a']))
            opt_a = self.task.get_opt_action()
            next_states, rewards, terminals, next_infos = self.task.step(opt_a)
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
                         'ns': tensor(next_states),
                         'info': infos,
                         'opt_a': tensor(opt_a, dtype=torch.long)})

            states = next_states
            infos = next_infos

        self.states = states
        self.infos = infos
        prediction = self.network(states, infos)
        storage.add(prediction)
        storage.placeholder()

        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            storage.ret[i] = returns.detach()

        loss_dict = dict()
        states, next_states, infos, opt_a = storage.cat(['s', 'ns', 'info', 'opt_a'])
        log_prob = self.network.get_logprobs(states, infos)
        loss_dict['NLL'] = F.nll_loss(log_prob, opt_a)
        if hasattr(config, 'trans'):
            loss_dict['trans'] = config.trans.loss(states, opt_a, next_states)
        for k, v in loss_dict.items():
            config.logger.add_scalar(tag=k, value=v, step=self.total_steps)

        self.opt.step(sum(loss_dict.values(), 0.0))

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.network.step()
