#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch.nn.functional as F
from .network_utils import *
from .network_bodies import *

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        return y

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi

class ActorCriticNet(nn.Module, BaseNet):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v}

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}

### tsa ###

### directly calculate embedding

# a network that output probability
# not sure whether this is a good practice, since I call forward instead of __call__
class ProbNet(VanillaNet):
    def __init__(self, output_dim, body):
        super().__init__(output_dim, body)

    def forward(self, x):
        y = super().forward(x)
        assert y.dim() == 2, 'output of ProbNet should be of dim 2'
        return nn.functional.softmax(y, dim=1)

class EmbeddingActorNet(nn.Module, BaseNet):
    def __init__(self, n_abs, action_dim, n_tasks):
        super().__init__()
        self.weights = weight_init(torch.randn(n_tasks, n_abs, action_dim)) # each col should be log_prob

    def forward(self, cs, info, action=None):
        assert cs.dim() == 2, 'dimension of cs should be 2' # N x C
        weights = self.weights[tensor(info['task_id'], torch.int64),:,:] # N x C x A
        weights = nn.functional.softmax(weights, dim=2)
        probs = torch.bmm(cs.unsqueeze(1), weights).squeeze(1) # Nx1xC @ NxCxA = Nx1xA -> NxA
        assert np.allclose(probs.detach().numpy().sum(1), np.ones(probs.size(0))) # change
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1) # unsqueeze!
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy}

class AttentionActorNet(EmbeddingActorNet):
    def __init__(self, n_abs, abs_dim, action_dim, n_tasks):
        super().__init__(n_abs, action_dim, n_tasks)
        self.abs_feats = weight_init(torch.randn(abs_dim, n_abs))

    def forward(self, xs, info, action=None):
        # xs: NxD, abs_feat: DxC
        cs = nn.functional.softmax(torch.matmul(F.normalize(xs), F.normalize(self.abs_feats, dim=0)), dim=1)
        return super().forward(cs, info, action=action)

# each task maintains a linear layer
# input: abstract feature
# output: action
class LinearActorNet(nn.Module, BaseNet):
    def __init__(self, abs_dim, action_dim, n_tasks):
        super().__init__()
        self.weights = nn.Parameter(weight_init(torch.randn(n_tasks, abs_dim, action_dim)))
        self.biases = nn.Parameter(torch.zeros(n_tasks, action_dim))

    def forward(self, xs, info, action=None):
        weights = self.weights[tensor(info['task_id'], torch.int64),:,:]
        biases = self.biases[tensor(info['task_id'], torch.int64),:]
        output = torch.bmm(xs.unsqueeze(1), weights).squeeze(1) + biases
        probs = nn.functional.softmax(output, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1) # unsqueeze!
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy}

class TSAActorCriticNet(nn.Module, BaseNet):
    def __init__(self, action_dim, phi, actor, critic):
        super().__init__()
        self.phi = phi
        self.actor = actor
        self.critic = critic

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())
        self.phi_params = list(self.phi.parameters())

class VQNet(nn.Module, BaseNet):
    def __init__(self, embed_dim, body, n_embed):
        super().__init__()
        self.body = body
        self.embed_fc = layer_init(nn.Linear(body.feature_dim, embed_dim))
        self.embed = torch.nn.Embedding(n_embed, embed_dim) # weight shape: n_embed, embed_dim
    
    def forward(self, xs):
        xs = self.body(xs)
        distance = (xs ** 2).sum(dim=1, keepdim=True) + (self.embed.weight ** 2).sum(1) - 2 * torch.matmul(xs, self.embed.weight.t())
        indices = torch.argmin(distance, dim=1)
        print('# of indices:', len(set(indices.detach().numpy())))
        output = self.embed(indices)
        output_x = xs + (output - xs).detach()
        e_latent_loss = torch.mean((output.detach() - xs)**2)
        q_latent_loss = torch.mean((output - xs.detach())**2)
        self._loss = q_latent_loss + 0.25 * e_latent_loss

        return output_x # output the one that can pass gradient to xs

class CategoricalTSAActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 phi, # state |-> abstract state
                 actor, # abstract state |-> action
                 critic): # state |-> value function
        super().__init__()
        self.network = TSAActorCriticNet(action_dim, phi, actor, critic)
        self.to(Config.DEVICE)

    def forward(self, obs, info, action=None):
        obs = tensor(obs)
        abs_s = self.network.phi(obs) # abstract state
        output = self.network.actor(abs_s, info, action=action)
        output['v'] = self.network.critic(obs)
        return output

### end of tsa ###
