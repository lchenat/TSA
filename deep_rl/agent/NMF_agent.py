from ..network import *
from ..component import *
from collections import defaultdict
from .BaseAgent import NMFBaseAgent

import numpy as np


class NMFAgent(NMFBaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network)
        self.total_steps = 0
        self.policy_loss = torch.nn.KLDivLoss()
        self.abs_loss = torch.nn.MSELoss() if config.abs_mode == 'mse' else torch.nn.KLDivLoss()

        self.abs = np.concatenate([config.sample_dict['abs'] for _ in range(len(config.sample_dict['states']))])
        if config.abs_mode == 'mse' and config.abs_mean is not None:
            print('multiplied by: {}'.format(config.abs_mean / self.abs.mean()))
            self.abs *= config.abs_mean / self.abs.mean()
        if config.abs_mode == 'kl':
            assert np.allclose(self.abs.sum(1), 1.0)
        self.states = np.concatenate(config.sample_dict['states'])
        self.infos = np.concatenate(config.sample_dict['infos'])
        #self.abs = np.concatenate([config.sample_dict['abs'] for _ in range(len(self.states) // len(config.sample_dict['abs']))])
        self.policies = np.concatenate(config.sample_dict['policies'])
 
    def eval_step(self, state, info):
        action = self.network(state, info)['a'][0].cpu().detach().numpy()
        if not action.shape: action = action.item()
        return action

    def eval_episode(self):
        config = self.config
        env = self.config.eval_env
        state = config.state_normalizer(env.reset())
        info = env.get_info()
        total_rewards = 0
        while True:
            action = self.eval_step(state, info)
            state, reward, done, _ = env.step([action])
            state = config.state_normalizer(state)
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
        indices = np.random.choice(len(self.states), size=config.batch_size)
        states = tensor(config.state_normalizer(self.states[indices]))
        infos = stack_dict(self.infos[indices]) # infos should be list of dictionary!
        expected_abs = tensor(self.abs[indices])
        expected_policies = tensor(self.policies[indices])
        if hasattr(self.network, 'abs_encoder'):
            if config.abs_mode == 'mse':
                actual_abs = self.network.abs_encoder(states, infos)
            else: 
                actual_abs = self.network.abs_encoder.get_logprob(states, infos)
        else:
            assert config.abs_mode == 'mse'
            actual_abs = self.network.network.phi_body(states)
        actual_policies = F.log_softmax(self.network.get_logits(states, infos), dim=-1)
        loss_dict = dict()
        #loss_dict['NLL'] = (-logprobs * labels).sum(dim=1).mean()
        # this is backward compatible
        loss_dict['KL'] = self.policy_loss(actual_policies.view(-1, actual_policies.shape[-1]), expected_policies.view(-1, expected_policies.shape[-1]))
        loss_dict['abs_loss'] = self.abs_loss(actual_abs, expected_abs)
        loss_dict['network'] = self.network.loss()
        for loss in loss_dict.values(): assert loss == loss, 'NaN detected'
        # log before update
        loss = config.kl_coeff * loss_dict['KL'] + config.abs_coeff * loss_dict['abs_loss']
        self.loss = loss.detach().cpu().numpy()
        for k, v in loss_dict.items():
            config.logger.add_scalar(tag=k, value=v, step=self.total_steps)
        self.opt.step(loss)
        self.total_steps += config.batch_size
        self.network.step() # do all adaptive update
