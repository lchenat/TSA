from ..network import *
from ..component import *
from .BaseAgent import *

from collections import defaultdict


class TransferAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.source = config.source_fn()
        self.target = config.target_fn()
        self.source_opt = config.optimizer_fn(self.source)
        self.target_opt = config.optimizer_fn(self.target)
        self.total_steps = 0
        self.online_rewards = np.zeros(config.num_workers)
        self.episode_rewards = []
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.infos = self.task.get_info()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        infos = self.infos
        for t in range(config.rollout_length):
            prediction = self.target(states, infos)
            source_log_pi_a = self.source(states, infos, action=prediction['a'])['log_pi_a']
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
                         'info': infos,
                         'source_log_pi_a': source_log_pi_a,
                         })
            states = next_states
            infos = next_infos

        self.states = states
        self.infos = infos
        prediction = self.target(states, infos) # what is this for? Just fill in blank?
        storage.add(prediction)
        storage.placeholder() # fill in keys with no value

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        #discount = tensor(np.ones((config.num_workers, 1))) # discount sum of log_probability
        for i in reversed(range(config.rollout_length)):
            #returns = storage.r[i] + config.distill_w * storage.source_log_pi_a[i] + config.discount * storage.m[i] * returns
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            #storage.discount_source_log_pi_a[i] = discount * storage.source_log_pi_a[i]
            #discount = discount * storage.m[i] * config.discount + (1 - storage.m[i])
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                #td_error = storage.r[i] + config.distill_w * storage.source_log_pi_a[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]

                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        # advantages <- adv
        # cat data from all workers together (mix)
        #states, next_states, ms, actions, log_probs_old, returns, advantages, infos, discount_source_log_pi_a = storage.cat(['s', 'ns', 'm', 'a', 'log_pi_a', 'ret', 'adv', 'info', 'discount_source_log_pi_a'])
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

                # target loss
                prediction = self.target(sampled_states, sampled_infos, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()
                loss_dict['policy'].append(policy_loss)
                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()
                loss_dict['value'].append(value_loss)
                network_loss = self.target.loss() # maybe include both
                loss_dict['target_network'].append(network_loss)
                aux_loss = network_loss
                if getattr(config, 'action_predictor', None) is not None: # inverse dynamic loss
                    indices = sampled_ms.squeeze(1) > 0
                    action_prediction_loss = 0.05 * config.action_predictor.loss(sampled_states[indices], sampled_next_states[indices], sampled_actions[indices])
                    loss_dict['action'].append(action_prediction_loss)
                    aux_loss += action_prediction_loss
                if getattr(config, 'recon', None) is not None:
                    recon_loss = config.recon.loss(sampled_states)
                    loss_dict['recon'].append(recon_loss)
                    aux_loss += recon_loss
                ### optimization ###
                self.target_opt.step(policy_loss + value_loss + aux_loss, retain_graph=True)

        # source loss
        source_prob = self.source.get_probs(states, infos)
        target_prob = self.target.get_probs(states, infos)
        target_loss = config.distill_w * F.kl_div(target_prob, source_prob.detach())
        source_loss = config.distill_w * F.kl_div(source_prob, target_prob.detach())
        self.source_opt.step(source_loss, retain_graph=True)
        self.target_opt.step(target_loss)

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        for k, v in loss_dict.items():
            config.logger.add_scalar(tag=k, value=torch.mean(tensor(v)), step=self.total_steps)
        self.source.step()
        self.target.step() # do all adaptive update
        
