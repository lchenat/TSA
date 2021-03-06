#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from ..utils.torch_utils import *

from collections import defaultdict


class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network)
        self.total_steps = 0
        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.infos = self.task.get_info() # store task_ids
        #check_network_output(self) # debug

    def eval_step(self, state, info):
        action = self.network(state, info)['a'].cpu().detach().numpy()
        return action

    def eval_episode(self):
        config = self.config
        env = self.config.eval_env
        state = config.state_normalizer(env.reset())
        info = env.get_info()
        total_rewards = 0
        while True:
            action = self.eval_step(state, info)
            state, reward, done, info = env.step(action)
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
        storage = Storage(config.rollout_length)
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
                         'ns': tensor(next_states),
                         'info': infos}) # cat?
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
        states, next_states, ms, actions, log_probs_old, returns, advantages, infos = storage.cat(['s', 'ns', 'm', 'a', 'log_pi_a', 'ret', 'adv', 'info'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()
        loss_dict = defaultdict(list) # record loss and output

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_next_states = next_states[batch_indices]
                sampled_ms = ms[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                sampled_infos = {k: [v[n] for n in to_np(batch_indices)] for k, v in infos.items()}

                prediction = self.network(sampled_states, sampled_infos, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()
                loss_dict['policy'].append(policy_loss)
                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()
                loss_dict['value'].append(value_loss)
                network_loss = self.network.loss()
                #network_loss = network_loss / torch.clamp(network_loss.detach(), min=1)
                loss_dict['network'].append(network_loss)
                aux_loss = network_loss
                if hasattr(config, 'reg_abs'):
                    reg_abs_loss = config.reg_abs.loss(sampled_states, sampled_infos)
                    loss_dict['reg_abs'].append(reg_abs_loss)
                ### optimization ###
                self.opt.step(policy_loss + value_loss + aux_loss) # try first
        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        # log the abs_encoder
        if self.task.env_type == 'full':
            env = config.eval_env.env.envs[0]
            all_states, all_infos = get_states_infos(env, config.discount)
            all_states = tensor(config.state_normalizer(all_states))
            all_infos = stack_dict(all_infos) 
        elif self.task.env_type == 'simulation':
            all_states = states
            all_infos = infos
        if hasattr(self.network, 'abs_encoder'): # temp forbided for sample
            if self.network.abs_encoder.abstract_type == 'sample':
                indices = self.network.abs_encoder.get_indices(all_states, all_infos).detach().cpu().numpy()
                indices = [tuple(index) for index in indices]
                i2e, e2i = index_dict(indices)
                indices = [e2i[index] for index in indices]
            else:
                indices = self.network.abs_encoder.get_indices(all_states, all_infos).detach().cpu().numpy()
            if 'pos' in all_infos:
                abs_map = {pos: i for pos, i in zip(all_infos['pos'], indices)}
                config.logger.add_scalar(tag='n_abs', value=len(set(abs_map.values())), step=self.total_steps)
                config.logger.add_file('abs_map', abs_map, step=self.total_steps)
            # log the visualization of the abs
            if hasattr(self.network.abs_encoder, 'abstract_type'):
                if self.network.abs_encoder.abstract_type == 'max':
                    n_used_indices = len(set(self.network.abs_encoder.get_indices(states, infos).detach().cpu().numpy()))
                    config.logger.add_scalar(tag='n_used_abstract_states', value=n_used_indices, step=self.total_steps)
                elif self.network.abs_encoder.abstract_type == 'prob':
                    entropy = self.network.abs_encoder.entropy(states, infos)
                    config.logger.add_scalar(tag='UB_on_abstract_states', value=torch.exp(entropy).mean().detach().cpu().numpy(), step=self.total_steps)
                elif self.network.abs_encoder.abstract_type == 'sample':
                    pass
        else: # count by clipping, categoricalactorcriticnet
            indices = (self.network.network.phi_body(all_states).detach().cpu().numpy() * 1e5).astype(int)
            indices = [tuple(index) for index in indices]
            config.logger.add_scalar(tag='n_abs', value=len(set(indices)), step=self.total_steps)
        for k, v in loss_dict.items():
            config.logger.add_scalar(tag=k, value=torch.mean(tensor(v)), step=self.total_steps)
        self.network.step() # do all adaptive update

