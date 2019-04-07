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
        return np.mean(rewards)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        infos = self.infos
        for _ in range(config.rollout_length):
            prediction = self.network(states, infos)
            next_states, rewards, terminals, next_infos = self.task.step(to_np(prediction['a'])) # follow current policy instead of optimal
            if config.expert is None:
                opt_a = self.task.get_opt_action()
            else:
                opt_a = [to_np(config.expert[j](states[i:i+1], {'task_id': [j]})['a']) for i, j in enumerate(infos['task_id'])] 
                opt_a = np.concatenate(opt_a)
            #next_states, rewards, terminals, next_infos = self.task.step(opt_a)
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
        if hasattr(self.network, 'abs_encoder'): # temp forbided for sample
            if self.network.abs_encoder.abstract_type == 'sample':
                indices = self.network.abs_encoder.get_indices(states, infos).detach().cpu().numpy()
                indices = [tuple(index) for index in indices]
                i2e, e2i = index_dict(indices)
                indices = [e2i[index] for index in indices]
            else:
                indices = self.network.abs_encoder.get_indices(states, infos).detach().cpu().numpy()
            abs_map = {pos: i for pos, i in zip(infos['pos'], indices)}
            config.logger.add_scalar(tag='n_abs', value=len(set(abs_map.values())), step=self.total_steps)
            config.logger.add_file('abs_map', abs_map, step=self.total_steps)
        self.network.step()
