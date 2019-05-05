from ..network import *
from ..component import *
from .BaseAgent import *
from ..utils import *

def roll(x):
    return torch.cat([x[1:], x[:1]])

def fitted_q(data, body, feat_dim, action_dim, discount):
    A = torch.zeros(feat_dim * action_dim + action_dim, feat_dim * action_dim + action_dim)
    b = torch.zeros(feat_dim * action_dim + action_dim)
    N = len(data['states'])

    feats = body(tensor(data['states'])).repeat(1, action_dim)
    #a_vec = one_hot.encode(tensor(data['actions'], torch.long), action_dim)
    #phis = torch.cat([feats * a_vec.repeat_interleave(feat_dim, 1), a_vec], 1)
    a_vec = np.eye(action_dim)[data['actions']]
    phis = torch.cat([feats * tensor(a_vec.repeat(feat_dim, 1)), tensor(a_vec)], 1)
    
    A = torch.matmul(phis.t(), phis - discount * tensor(1 - data['terminals']).unsqueeze(1) * roll(phis)) / N
    b = torch.matmul(phis.t(), tensor(data['rewards'])) / N
    
    # update weight
    total_weight = torch.matmul(torch.inverse(A + 1e-4 * tensor(np.eye(A.shape[0]))), b)
    weight = total_weight[:-action_dim].view(-1, feat_dim).t()
    bias = total_weight[-action_dim:]
    return weight, bias

class MLQAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network)
        self.total_steps = 0
        self.replay = config.replay_fn()
        self.states = config.state_normalizer(self.task.reset()) # change
        self.infos = self.task.get_info() # store task_ids

        self.episode_rewards = []
        self.online_rewards = np.zeros(config.num_workers)
        self.fc = nn.Linear(self.network.body.feature_dim, config.action_dim)
        self.fc.to(Config.DEVICE)

    def eval_step(self, state, info):
        return self.network(state, info)[0].argmax()

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
        return 0.0
        #rewards = []
        #for ep in range(self.config.eval_episodes):
            #rewards.append(self.eval_episode())
        #self.config.logger.info('evaluation episode return: %f(%f)' % (
            #np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))))
        #self.config.logger.add_scalar(tag='eval_return', value=np.mean(rewards), step=self.total_steps)
        #return np.mean(rewards)

    def step(self):
        config = self.config
        states = self.states
        infos = self.infos
        for _ in range(config.rollout_length):
            #prediction = self.network(states, infos)
            #next_states, rewards, terminals, next_infos = self.task.step(to_np(prediction['a'])) # follow current policy instead of optimal
            expert_q = [to_np(config.expert[j](states[i:i+1], {'task_id': [j]})[0]) for i, j in enumerate(infos['task_id'])]
            opt_a = [q.argmax() for q in expert_q]
            #storage.add({'expert_q': tensor(np.concatenate(expert_q))})
            next_states, rewards, terminals, next_infos = self.task.step(opt_a)
            self.online_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.episode_rewards.append(self.online_rewards[i])
                    self.online_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            #storage.add(prediction)
            #storage.add({'r': tensor(rewards).unsqueeze(-1),
                         #'m': tensor(1 - terminals).unsqueeze(-1),
                         #'s': tensor(states),
                         #'ns': tensor(next_states),
                         #'info': infos,
                         #'opt_a': tensor(opt_a, dtype=torch.long)})
            if self.replay.size() < 100:
                self.replay.feed(
                    [states,
                    opt_a,
                    rewards,
                    next_states,
                    terminals,
                    expert_q],
                    terminals,
                )

            states = next_states
            infos = next_infos

        self.states = states
        self.infos = infos
        #prediction = self.network(states, infos)
        #storage.add(prediction)
        #storage.placeholder()

        #returns = prediction['v'].detach()
        #for i in reversed(range(config.rollout_length)):
            #returns = storage.r[i] + config.discount * storage.m[i] * returns
            #storage.ret[i] = returns.detach()

        if self.replay.size() < 2 * self.replay.batch_size: return
        loss_dict = dict()
        #states, next_states, infos, opt_a, expert_q = storage.cat(['s', 'ns', 'info', 'opt_a', 'expert_q'])
        states, actions, rewards, _, terminals, expert_q = self.replay.sample()
        data = dict(
            states=states,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            expert_q=expert_q,
        )       
        weight, bias = fitted_q(data, self.network.body, self.network.body.feature_dim, config.action_dim, config.discount)
        self.fc.weight.data.copy_(weight.t()) # currently only support single task
        self.fc.bias.data.copy_(bias)
        estimate_q = self.fc(self.network.body(tensor(data['states'])))
        loss_dict['MSE'] = F.mse_loss(estimate_q, tensor(data['expert_q']))

        for k, v in loss_dict.items():
            config.logger.add_scalar(tag=k, value=v, step=self.total_steps)

        self.opt.step(sum(loss_dict.values(), 0.0))

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.network.step()
