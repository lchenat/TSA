from ..network import *
from ..component import *
from .BaseAgent import *

from collections import defaultdict

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = torch.sort(v, descending=True)[0]
    cssv = torch.cumsum(u, 0) - z
    ind = tensor(np.arange(n_features) + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho.float()
    #w = np.maximum(v - theta, 0)
    w = F.relu(v - theta)
    return w

def update_v(X, U, V):
    K = U.shape[1]
    Y = X - torch.matmul(U, V) + torch.ger(U[:, 0], V[0, :]) # Y_1
    for k in range(K):
        #Y = X - torch.matmul(U, V) + torch.ger(U[:, k], V[k, :])
        V[k, :] = projection_simplex_sort(torch.matmul(Y.t(), U[:, k]) / torch.dot(U[:, k], U[:, k]))
        if k < K-1:
            Y = Y - torch.ger(U[:, k], V[k, :]) + torch.ger(U[:, k+1], V[k+1, :])

# X: (N, S, A)
# U: (S, K)
# V: (N, K, A)
def batch_update_v(X, U, V):
    K = U.shape[1]
    Us = U.unsqueeze(0).expand(V.shape[0], -1, -1)
    Y = X - torch.bmm(Us, V) + torch.bmm(Us, V)
    for k in range(K):
        #Y = X - torch.matmul(U, V) + torch.ger(U[:, k], V[k, :])
        V[k, :] = projection_simplex_sort(torch.matmul(Y.t(), U[:, k]) / torch.dot(U[:, k], U[:, k]))
        if k < K-1:
            Y = Y - torch.ger(U[:, k], V[k, :]) + torch.ger(U[:, k+1], V[k+1, :])   

class NMFDirectAgent(BaseAgent):
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
        #return self.network(state, info)['a'][0]
        return self.network.get_logits(state, info).max(dim=1)[1]

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

    def get_u(self, states):
        if hasattr(self.network, 'abs_encoder'):
            U = self.network.abs_encoder(states, None)
        else:
            U = self.network.network.phi_body(states)
        return U

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        infos = self.infos
        for _ in range(config.rollout_length):
            prediction = self.network(states, infos)
            #next_states, rewards, terminals, next_infos = self.task.step(to_np(prediction['a'])) # follow current policy instead of optimal
            if config.expert is None:
                opt_a = self.task.get_opt_action()
            else:
                opt_a = [to_np(config.expert[j](states[i:i+1], {'task_id': [j]})['a']) for i, j in enumerate(infos['task_id'])] 
                opt_a = np.concatenate(opt_a)
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

        loss_dict = defaultdict(list)
        if hasattr(self, 'replay'): # debug
            states = self.replay['states']
            next_states = self.replay['next_states']
            infos = self.replay['infos']
            opt_a = self.replay['opt_a']
        else:
            states, next_states, infos, opt_a = storage.cat(['s', 'ns', 'info', 'opt_a'])
            self.replay = dict(
                states=states,
                next_states=next_states,
                infos=infos,
                opt_a=opt_a,
            )
        # loss
        # define loss function
        def get_loss(Xs, U, Vs):
            X = torch.stack([Xs[i] for i in config.expert]).detach()
            V = torch.stack([Vs[i] for i in config.expert]).detach() # detach is important
            return F.mse_loss(torch.bmm(U.unsqueeze(0).expand(V.shape[0], *U.shape), V), X)
        ###
        Xs = dict()
        Vs = dict()
        for i, expert in config.expert.items():
            Xs[i] = F.softmax(expert.get_logits(states, {'task_id': [i] * len(states)}), dim=-1)
            if hasattr(self.network, 'abs_encoder'): # abs_encoder is not working very well now
                Vs[i] = self.network.actor.get_weight({'task_id': [i]}).squeeze(0)
            else:
                Vs[i] = self.network.network.actor.get_weight({'task_id': [i]}).squeeze(0)
        for i in range(config.x_iter):
            # update u
            for _ in range(config.u_iter):
                U = self.get_u(states)
                loss_dict['u_loss'].append(get_loss(Xs, U, Vs))
                self.opt.step(loss_dict['u_loss'][-1]) # not sure whether this work, since V has no gradient
            # update v
            U = self.get_u(states)
            for _ in range(config.v_iter):
                for i in config.expert:
                    update_v(Xs[i], U, Vs[i])
            loss_dict['v_loss'].append(get_loss(Xs, U, Vs))
        if hasattr(self.network, 'abs_encoder'):
            self.network.actor.load_weight(Vs)
        else:
            self.network.network.actor.load_weight(Vs) # not support abs_encoder yet

        for k, v in loss_dict.items():
            val = torch.mean(tensor(v))
            config.logger.add_scalar(tag=k, value=val, step=self.total_steps)
            config.logger.info('{}: {}'.format(k, val))

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.network.step()
