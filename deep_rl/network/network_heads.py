#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch.nn as nn
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

    def forward(self, obs, info, action=None):
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

### a simple baseline ###
# a network that output probability
class ProbAbstractEncoder(VanillaNet):
    def __init__(self, n_abs, body, abstract_type='prob'):
        super().__init__(n_abs, body)
        self.abstract_type='prob'
        self.loss_weight = 0.001

    def get_indices(self, inputs, info):
        y = super().forward(x)
        return torch.argmax(y, dim=1)

    def forward(self, x, info):
        y = super().forward(x)
        self._loss = self.loss_weight * self.entropy(x, info, logits=y).mean()
        return nn.functional.softmax(y, dim=1)

    def entropy(self, x, info, logits=None):
        if logits is None:
            logits = self.forward(x, info)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy()

# maintain embeddings for abstract state
class EmbeddingActorNet(nn.Module, BaseNet):
    def __init__(self, n_abs, action_dim, n_tasks):
        super().__init__()
        self.weight = nn.Parameter(weight_init(torch.randn(n_tasks, n_abs, action_dim)))

    def forward(self, cs, info, action=None):
        assert cs.dim() == 2, 'dimension of cs should be 2'
        weights = nn.functional.softmax(self.weight[tensor(info['task_id'], torch.int64),:,:], dim=2)
        probs = batch_linear(cs, weight=weights)
        assert np.allclose(probs.detach().cpu().numpy().sum(1), np.ones(probs.size(0)))
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1) # unsqueeze!
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy}

### directly calculate embedding

class AbstractedStateEncoder(nn.Module, BaseNet):
    def __init__(self, n_embed, embed_dim, body, abstract_type='max'):
        super().__init__()
        self.aux_weight = 1
        self.temperature = 0.1
        self.abstract_type = abstract_type

        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.abs_states = nn.Parameter(weight_init(torch.randn(n_embed, embed_dim)))

    def forward(self, x):
        z = self.fc_head(F.relu(self.body(tensor(x))))
        abs_loss = 0
        if self.abstract_type == 'max':
            normalized_states = F.normalize(self.abs_states)
            normalized_z = F.normalize(z)
            quantization_score = F.softmax(F.linear(normalized_z, normalized_states) / self.temperature, dim=1) # probability quantization
            abs_ind = quantization_score.argmax(dim=1)
            # print(set(to_np(abs_ind).tolist()))

            abs_state = F.embedding(abs_ind, normalized_states)
            compatible_scores = F.softmax(F.linear(normalized_z, abs_state) / self.temperature, dim=1)
            abs_loss += F.cross_entropy(compatible_scores, diag_gt(compatible_scores))

            # Regularization#1: Diagnalize normalized similairity matrix
            abs_sim = F.softmax(F.linear(normalized_states, normalized_states) / self.temperature, dim=1)
            abs_loss += F.cross_entropy(abs_sim, diag_gt(abs_sim))
        else:
            raise ValueError('Only support max out for now')

        self._loss = self.aux_weight * abs_loss
        return abs_state

# each task maintains a linear layer
# input: abstract feature
# output: action
class LinearActorNet(MultiLinear, BaseNet):
    def __init__(self, abs_dim, action_dim, n_tasks):
        super().__init__(abs_dim, action_dim, n_tasks, 'task_id')

    def forward(self, xs, info, action=None):
        output = super().forward(xs, info)
        probs = nn.functional.softmax(output, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1) # unsqueeze!
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy}

class NonLinearActorNet(nn.Module, BaseNet):
    def __init__(self, abs_dim, action_dim, n_tasks, hidden_dim=512):
        super().__init__()
        self.fc1 = MultiLinear(abs_dim, hidden_dim, n_tasks, 'task_id')
        self.fc2 = MultiLinear(hidden_dim, action_dim, n_tasks, 'task_id')

    def forward(self, xs, info, action=None):
        output = F.relu(self.fc1(xs, info))
        output = self.fc2(output, info)
        probs = nn.functional.softmax(output, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1) # unsqueeze!
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy}

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

        output = batch_linear(xs, weights, bias=biases)
        probs = F.softmax(output, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1) # unsqueeze!
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy}

class VQAbstractEncoder(nn.Module, BaseNet):
    def __init__(self, n_embed, embed_dim, body, abstract_type='max'):
        super().__init__()
        self.body = body
        self.embed_fc = layer_init(nn.Linear(body.feature_dim, embed_dim))
        self.embed = torch.nn.Embedding(n_embed, embed_dim) # weight shape: n_embed, embed_dim
        self.abstract_type = abstract_type
        self.loss_weight = 0.01
    
    def get_features(self, inputs):
        return self.body(inputs)

    def get_indices(self, inputs_or_xs, info, is_feature=False):
        if not is_feature:
            xs = self.get_features(inputs_or_xs)
        else:
            xs = inputs_or_xs
        distance = (xs ** 2).sum(dim=1, keepdim=True) + (self.embed.weight ** 2).sum(1) - 2 * torch.matmul(xs, self.embed.weight.t())
        if self.abstract_type == 'max':
            indices = torch.argmin(distance, dim=1)
        elif self.abstract_type == 'softmax':
            indices = torch.distributions.Categorical(logits=distance).sample()
        return indices

    def get_embeddings(self, indices, xs=None):
        output = self.embed(indices)
        if xs is not None:
            output_x = xs + (output - xs).detach()
            return output, output_x
        else:
            return output

    def forward(self, inputs, info):
        xs = self.get_features(inputs)
        indices = self.get_indices(xs, info, is_feature=True)
        output, output_x = self.get_embeddings(indices, xs)
        e_latent_loss = torch.mean((output.detach() - xs)**2)
        q_latent_loss = torch.mean((output - xs.detach())**2)
        self._loss = self.loss_weight * (q_latent_loss + 0.25 * e_latent_loss)

        return output_x # output the one that can pass gradient to xs

# input: abstract dictionary
class PosAbstractEncoder(nn.Module, BaseNet):
    def __init__(self, n_abs, abs_dict):
        super().__init__()
        self.n_abs = n_abs
        self.abs_dict = abs_dict

    def get_indices(inputs, info):
        indices = tensor([self.abs_dict[map_id][pos] for map_id, pos in zip(info['map_id'], info['pos'])], dtype=torch.long)
        return indices

    def forward(self, inputs, info):
        c_indices = [self.abs_dict[map_id][pos] for map_id, pos in zip(info['map_id'], info['pos'])]
        cs = one_hot.encode(tensor(c_indices, dtype=torch.long), dim=self.n_abs)
        return cs

class TSANet(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 abs_encoder, # state |-> abstract state
                 actor, # abstract state |-> action
                 critic): # state |-> value function
        super().__init__()
        self.abs_encoder = abs_encoder
        self.actor = actor
        self.critic = critic

        self.abs_encoder_params = list(self.abs_encoder.parameters())
        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())

        self.to(Config.DEVICE)

    def forward(self, obs, info, action=None):
        obs = tensor(obs)
        abs_s = self.abs_encoder(obs, info) # abstract state
        output = self.actor(abs_s, info, action=action)
        output['v'] = self.critic(obs, info)
        return output

class TSACriticNet(nn.Module, BaseNet):
    def __init__(self, body, n_tasks):
        super().__init__()
        self.body = body
        self.fc = MultiLinear(body.feature_dim, 1, n_tasks, 'task_id')

    def forward(self, inputs, info):
        return self.fc(self.body(inputs), info)

class ActionPredictor(nn.Module):
    def __init__(self, action_dim, state_encoder, hidden_dim=256):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_predictor = nn.Sequential(
            nn.Linear(2 * self.state_encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1),
        )
        self.to(Config.DEVICE)

    def forward(self, states, next_states):
        state_features = self.state_encoder(states)
        next_state_features = self.state_encoder(next_states)
        inputs = torch.cat([state_features, next_state_features], dim=1)
        return self.action_predictor(inputs)

    def loss(self, states, next_states, actions):
        predicted_actions = self(states, next_states)
        return F.cross_entropy(predicted_actions, actions)

### end of tsa ###
