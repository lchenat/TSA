#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

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

class ActorCriticNet(nn.Module, BaseNet):
    def __init__(self, n_tasks, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = MultiLinear(actor_body.feature_dim, action_dim, n_tasks, key='task_id', w_scale=1e-3)
        self.fc_critic = MultiLinear(critic_body.feature_dim, 1, n_tasks, key='task_id', w_scale=1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

### tsa ###

class AbstractEncoder(ABC):
    @abstractmethod
    def get_indices(self, inputs, info):
        pass

    @abstractmethod
    def forward(self, inputs, info):
        pass

# a network that output probability
class ProbAbstractEncoder(VanillaNet, AbstractEncoder):
    def __init__(self, n_abs, body, temperature, abstract_type='prob'):
        super().__init__(n_abs, body)
        self.abstract_type = abstract_type
        self.loss_weight = 0.0
        self.feature_dim = n_abs # output_dim
        self.temperature = temperature
        next(temperature)

    def get_indices(self, inputs, info):
        y = super().forward(inputs)
        return torch.argmax(y, dim=1)

    def forward(self, inputs, info):
        y = super().forward(inputs) / self.temperature.cur
        self._loss = self.loss_weight * self.entropy(inputs, info, logits=y).mean() if self.loss_weight else 0.0
        return F.log_softmax(y, dim=1)
        #return F.softmax(y, dim=1) # debug

    def entropy(self, inputs, info, logits=None):
        if logits is None:
            logits = self.forward(inputs, info)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy()

    def step(self):
        next(self.temperature)
        BaseNet.step(self)

# gumbel softmax sampling
class SampleAbstractEncoder(VanillaNet, AbstractEncoder):
    def __init__(self, n_abs, body, temperature, base=2, abstract_type='sample'):
        assert n_abs % base == 0
        super().__init__(n_abs, body)
        self.abstract_type = abstract_type
        self.feature_dim = n_abs
        self.temperature = temperature
        next(temperature)
        self.base = base

    def get_indices(self, inputs, info):
        y = super().forward(inputs).view(inputs.size(0), -1, self.base)
        return torch.argmax(y, dim=2) # this indices it not the same as the others

    def forward(self, inputs, info):
        y = super().forward(inputs).view(inputs.size(0), -1, self.base)
        out = gumbel_softmax.hard_sample(y, self.temperature.cur)
        return out.view(y.size(0), -1)

    def step(self):
        next(self.temperature)
        BaseNet.step(self)

class BernoulliAbstractEncoder(VanillaNet, AbstractEncoder):
    def __init__(self, n_abs, body, temperature, abstract_type='sample'):
        super().__init__(n_abs, body)
        self.abstract_type = abstract_type
        self.feature_dim = n_abs
        self.temperature = temperature
        next(temperature)

    def get_indices(self, inputs, info):
        return self.forward(inputs, info)

    def forward(self, inputs, info):
        y = super().forward(inputs)
        return relaxed_Bernolli.hard_sample(y, self.temperature.cur) # this is sampling

    def step(self):
        next(self.temperature)
        BaseNet.step(self)

# use relaxed Bernoulli
class I2AAbstractEncoder(nn.Module, BaseNet, AbstractEncoder):
    def __init__(self, n_abs, body, temperature, feature_dim=512):
        super().__init__()
        self.abstract_type = 'sample'
        self.feature_dim = feature_dim
        self.temperature = temperature
        next(temperature)
        self.body = body
        self.mask_fc = nn.Linear(body.feature_dim, n_abs)
        self.feat_fc = nn.Linear(body.feature_dim, n_abs) # the feature dim along each abs is only 1 now

    def get_indices(self, inputs, info):
        xs = self.body(inputs)
        mask = relaxed_Bernolli.hard_sample(self.mask_fc(xs), self.temperature.cur)
        return mask

    def forward(self, inputs, info):
        xs = self.body(inputs)
        mask = relaxed_Bernolli.hard_sample(self.mask_fc(xs), self.temperature.cur)
        feat = self.feat_fc(xs)
        return mask * feat

    def step(self):
        next(self.temperature)
        BaseNet.step(self)

class VQAbstractEncoder(nn.Module, BaseNet, AbstractEncoder):
    def __init__(self, n_embed, embed_dim, body, abstract_type='max'):
        super().__init__()
        self.body = body
        self.embed_fc = layer_init(nn.Linear(body.feature_dim, embed_dim))
        self.embed = torch.nn.Embedding(n_embed, embed_dim) # weight shape: n_embed, embed_dim
        self.abstract_type = abstract_type
        self.loss_weight = 0.01
        self.feature_dim = embed_dim
    
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

class KVAbstractEncoder(nn.Module, BaseNet, AbstractEncoder):
    def __init__(self, n_embed, embed_dim, body, abstract_type='prob'):
        super().__init__()
        self.body = body
        #self.key = nn.Linear(body.feature_dim, n_embed, bias=False),
        self.key = nn.Linear(body.feature_dim, embed_dim)
        #self.value = nn.Linear(n_embed, embed_dim, bias=False)
        self.value = nn.Parameter(weight_init(torch.randn(n_embed, embed_dim)))
        self.abstract_type = abstract_type
        self.loss_weight = 0.0
        self.feature_dim = embed_dim
        self.denominator = np.sqrt(body.feature_dim)
    
    def entropy(self, inputs, info, logits=None):
        if logits is None:
            logits = self.get_logprobs(inputs, info)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy()

    def get_logprobs(self, inputs, info):
        key = self.key(self.body(inputs))
        return F.log_softmax(torch.matmul(F.normalize(key, dim=1), F.normalize(self.value, dim=1).t()), dim=1) 
        #return F.log_softmax(self.key(self.body(inputs)) / self.denominator, dim=1)

    def get_probs(self, inputs, info):
        key = self.key(self.body(inputs))
        return F.softmax(torch.matmul(F.normalize(key, dim=1), F.normalize(self.value, dim=1).t()), dim=1) 
        #return F.softmax(self.key(self.body(inputs)) / self.denominator, dim=1)

    def get_indices(self, inputs, info):
        return self.get_logprobs(inputs, info).argmax(dim=1)

    def forward(self, inputs, info):
        logprobs = self.get_logprobs(inputs, info)
        self._loss = self.loss_weight * self.entropy(inputs, info, logits=logprobs).mean()
        return torch.matmul(self.get_probs(inputs, info), self.value)

# input: abstract dictionary
class PosAbstractEncoder(nn.Module, BaseNet, AbstractEncoder):
    def __init__(self, n_abs, abs_dict):
        super().__init__()
        self.n_abs = n_abs
        self.abs_dict = abs_dict
        self.feature_dim = self.n_abs
        self.abstract_type = 'pos'

    def get_indices(self, inputs, info):
        indices = tensor([self.abs_dict[map_id][pos] for map_id, pos in zip(info['map_id'], info['pos'])], dtype=torch.long)
        return indices

    def forward(self, inputs, info):
        c_indices = [self.abs_dict[map_id][pos] for map_id, pos in zip(info['map_id'], info['pos'])]
        cs = one_hot.encode(tensor(c_indices, dtype=torch.long), dim=self.n_abs)
        return cs

####### Actor #######
class Actor(BaseNet):
    def get_logits(self, xs, info):
        raise NotImplementedError

    def get_logprobs(self, xs, info):
        return F.log_softmax(self.get_logits(xs, info), dim=-1)

    def get_probs(self, xs, info):
        return F.softmax(self.get_logits(xs, info), dim=-1)

class IdentityActor(nn.Module, BaseNet): # only works for single environment!
    def get_logits(self, x, info):
        return x

    def forward(self, x, info, action=None):
        dist = torch.distributions.Categorical(logits=x)
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
class LinearActorNet(MultiLinear, BaseNet):
    def __init__(self, abs_dim, action_dim, n_tasks):
        super().__init__(abs_dim, action_dim, n_tasks, key='task_id', w_scale=1e-3)

    def get_logits(self, xs, info):
        output = super().forward(xs, info)
        return output

    def forward(self, xs, info, action=None):
        #logprobs = self.get_logprobs(xs, info)
        #dist = torch.distributions.Categorical(logits=logprobs)
        dist = torch.distribution.Categorical(logits=self.get_logits(xs, info))
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
        self.fc1 = MultiLinear(abs_dim, hidden_dim, n_tasks, key='task_id', w_scale=1e-3)
        self.fc2 = MultiLinear(hidden_dim, action_dim, n_tasks, key='task_id', w_scale=1e-3)

    def get_logits(self, xs, info):
        output = F.relu(self.fc1(xs, info))
        output = self.fc2(output, info)
        return output

    def forward(self, xs, info, action=None):
        #logprobs = self.get_logprobs(xs, info)
        #dist = torch.distributions.Categorical(logits=logprobs)
        dist = torch.distributions.Categorical(logits=self.get_logits(xs, info))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1) # unsqueeze!
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy}

class TSACriticNet(nn.Module, BaseNet):
    def __init__(self, body, n_tasks):
        super().__init__()
        self.body = body
        self.fc = MultiLinear(body.feature_dim, 1, n_tasks, key='task_id', w_scale=1e-3)

    def forward(self, inputs, info):
        if isinstance(self.body, AbstractEncoder):
            return self.fc(self.body(inputs, info), info)
        return self.fc(self.body(inputs), info)

class CategoricalActorCriticNet(nn.Module, Actor):
    def __init__(self,
                 n_tasks,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(n_tasks, state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def get_logits(self, obs, info):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi) # maybe need info here, but not now
        logits = self.network.fc_action(phi_a, info)
        return logits

    def forward(self, obs, info, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi) # maybe need info here, but not now
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a, info)
        v = self.network.fc_critic(phi_v, info)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}

    def value(self, obs, info):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_v = self.network.critic_body(phi)
        v = self.network.fc_critic(phi_v, info)
        return v

class TSANet(nn.Module, Actor):
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

    def get_logits(self, obs, info):
        obs = tensor(obs)
        abs_s = self.abs_encoder(obs, info) # abstract state
        return self.actor.get_logits(abs_s, info)

    def forward(self, obs, info, action=None):
        obs = tensor(obs)
        abs_s = self.abs_encoder(obs, info) # abstract state
        output = self.actor(abs_s, info, action=action)
        output['v'] = self.critic(obs, info)
        return output

    def value(self, obs, info):
        obs = tensor(obs)
        return self.critic(obs, info)

### auxiliary networks

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

class UNetReconstructor(nn.Module):
    def __init__(self, encoder, in_channels):
        super().__init__()
        #self.encoder = UnetEncoder(in_channels)
        self.encoder = encoder
        self.decoder = UnetDecoder(in_channels)
        self.loss_weight = 1.0
        self.to(Config.DEVICE)

    def forward(self, states):
        return self.decoder(self.encoder(states))

    def loss(self, states):
        pred_states = self(states)
        return self.loss_weight * F.binary_cross_entropy(pred_states, states)

class TransitionModel(nn.Module):
    def __init__(self, encoder, action_dim, in_channels):
        super().__init__()
        self.encoder = encoder
        self.action_embed = nn.Embedding(action_dim, self.encoder.feature_dim)
        self.fusion_fc = nn.Linear(3*self.encoder.feature_dim, self.encoder.feature_dim)
        self.decoder = UnetDecoder(in_channels)
        self.loss_weight = 1.0
        self.to(Config.DEVICE)

    def forward(self, states, actions, next_states):
        state_features = self.encoder(states)
        next_state_features = self.encoder(next_states)
        action_features = self.action_embed(actions)
        features = torch.cat([state_features, next_state_features, action_features], dim=1)
        return self.decoder(F.relu(self.fusion_fc(features)))

    def loss(self, states, actions, next_states):
        return self.loss_weight * F.binary_cross_entropy(self(states, actions, next_states), (next_states - states + 1.0) / 2)
