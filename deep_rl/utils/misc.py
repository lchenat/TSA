#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import subprocess
import numpy as np
import pickle
import os
import datetime
import torch
import time
from .torch_utils import *
try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path

class with_null:
    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        return None

def stdin_choices(msg, choices, err_msg=None):
    choices_txt = '[{}]'.format('/'.join(choices))
    if err_msg is None:
        err_msg = 'choice not valid, please choose within {}'.format(choices_txt)
    while True:
        choice = input('{}{}:'.format(msg, choices_txt))
        if choice in choices:
            return choice
        print(err_msg)

# run git diff and return whether there is modification
def is_git_diff():
    return bool(subprocess.check_output(['git', 'diff']))

def lastname(path):
    return os.path.basename(os.path.normpath(path)) # might have problem for symbolic link

def collect_stats(agent):
    rewards = agent.episode_rewards
    return {
        'steps': agent.total_steps,
        'mean returns': np.mean(rewards),
        'median returns': np.median(rewards),
        'min returns': np.min(rewards),
        'max returns': np.max(rewards),
    }
   
def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            stats = collect_stats(agent)
            agent.episode_rewards = [] # clear stats
            config.logger.info('total steps {}, returns {:.2f}/{:.2f}/{:.2f}/{:.2f} (mean/median/min/max), {:.2f} steps/s'.format(
                stats['steps'], stats['mean returns'], stats['median returns'], stats['min returns'], stats['max returns'],
                config.log_interval / (time.time() - t0)))
            config.logger.add_scalar('mean-returns', stats['mean returns'], stats['steps'])
            t0 = time.time()
            if config.save_interval and not agent.total_steps % (config.save_interval * config.log_interval):
                agent.save('data/{}/step-{}-mean-{}' % (config.log_name, stats['steps'], stats['mean returns']))
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()

#def run_steps(agent):
#    config = agent.config
#    agent_name = agent.__class__.__name__
#    t0 = time.time()
#    while True:
#        if config.save_interval and not agent.total_steps % config.save_interval:
#            agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
#        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
#            rewards = agent.episode_rewards
#            agent.episode_rewards = []
#            config.logger.info('total steps %d, returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), %.2f steps/s' % (
#                agent.total_steps, np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards),
#                config.log_interval / (time.time() - t0)))
#            t0 = time.time()
#        if config.eval_interval and not agent.total_steps % config.eval_interval:
#            agent.eval_episodes()
#        if config.max_steps and agent.total_steps >= config.max_steps:
#            agent.close()
#            break
#        agent.step()

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()

def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

### tsa ###
def get_log_dir(name):
    if os.path.exists('./log/{}.txt'.format(name)):
        if stdin_choices('{} exists, do you want to replace?'.format(name), ['y', 'n']) == 'y':
            return './log/{}'.format(name)
        else:
            print('exit the program')
            exit()

# stack tree built by dictionary
def stack_dict(args, stack=None):
    ret = {}
    for k in args[0]:
        t = []
        for d in args:
            t.append(d[k])
        if isinstance(args[0][k], dict):
            ret[k] = stack_dict(t, stack)
        else:
            if stack:
                t = stack(t)
            ret[k] = t
    return ret

