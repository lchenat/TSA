#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .config import *
import torch
import os
import random
from torch import nn

def select_device(gpu_id):
    # if torch.cuda.is_available() and gpu_id >= 0:
    if gpu_id >= 0:
        Config.DEVICE = torch.device('cuda:%d' % (gpu_id))
    else:
        Config.DEVICE = torch.device('cpu')

def tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.type(dtype)
    x = torch.tensor(x, device=Config.DEVICE, dtype=dtype)
    return x

def tensor_dict(d, dtype=torch.float32):
    return {k: tensor(v, dtype=dtype) for k, v in d.items()}

def range_tensor(end):
    return torch.arange(end).long().to(Config.DEVICE)

def to_np(t):
    if torch.is_tensor(t):
        return t.cpu().detach().numpy()
    return t

def random_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

def epsilon_greedy(epsilon, x):
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)

def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()

def diag_gt(score_matrix):
    assert score_matrix.dim() == 2, 'score matrix needs dim = 2.'
    return torch.LongTensor(range(score_matrix.size(0))).to(score_matrix.device)

def batch_linear(input, weight, bias=None):
    """ input: (N, D), weight: (N, D, H), bias: (N, H) """
    if bias is not None:
        return torch.bmm(input.unsqueeze(1), weight).squeeze(1) + bias
    else:
        return torch.bmm(input.unsqueeze(1), weight).squeeze(1)

class one_hot:
    # input should be 1-dim indices or 2-dim with second dim 1
    # input should be LongTensor
    # output is FloatTensor
    @staticmethod
    def encode(indices, dim):
        if len(indices.shape) == 1:
            indices = indices.unsqueeze(1)
        assert len(indices.shape) == 2 and indices.shape[1] == 1, 'shape error'

        encodings = torch.zeros(indices.shape[0], dim).to(indices.device)
        encodings.scatter_(1, indices, 1)
        return encodings

    @staticmethod
    def decode(encodings):
        indices = encodings.nonzero()
        return indices[:, 1]

### optimizer ###
class VanillaOptimizer:
    def __init__(self, params, opt, grad_clip):
        self.params = params
        self.opt = opt # params already passed in
        self.grad_clip = grad_clip

    def step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, self.gradient_clip)
        self.opt.step()

# update the first / second params using the first / second opt with freq_list[0/1] times before switching
class AlternateOptimizer:
    def __init__(self, params_list, opt_list, freq_list, grad_clip):
        self.params_list = params_list
        self.opt_list = opt_list
        self.freq_list = freq_list
        self.grad_clip = grad_clip
        self.cur = 0 # current parameter to update
        self.t = 0 # count how many times the current parameter has been update
        
    def step(self, loss):
        opt = self.opt_list[self.cur]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params_list[self.cur], self.grad_clip)
        opt.step()
        self.t += 1
        if self.t >= self.freq_list[self.cur]:
            self.t = 0
            self.cur = 1 - self.cur
