#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import subprocess
import numpy as np
import pickle
import os
import git
import sys
import datetime
import filelock
import random
import torch
import time
import dill
import shlex
import shutil
import argparse
from PIL import Image
from .torch_utils import *
try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path


def permutation_num(n, k):
    res = 1
    for i in range(k):
        res *= n - i
    return res

# commandr
_cmd_dict = {} 

def cmd(name=None):
    def f(g):
        nonlocal name
        if name is None:
            name = g.__name__
        _cmd_dict[name] = g
        return g
    return f

def parse_args_as_func(argv):
    args = []
    kwargs = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith('-'):
            kwargs[argv[i].lstrip('-')] = argv[i+1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    return args, kwargs

def cmd_frun(name, *args, **kwargs):
    return _cmd_dict[name](*args, **kwargs)

def cmd_run(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args, kwargs = parse_args_as_func(argv)
    cmd_frun(args[0], *args[1:], **kwargs)


# synthesize name for logging
def sythesize_name(attr_dict, positional=[]):
    attrs = []
    for k, v in attr_dict.items():
        if k in positional:
            attrs.append(v)
        else:
            attrs.append('{}-{}'.format(k, v))
    return '.'.join(attrs)

def line_prepend(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content) # \r: return, \n: newline

def get_hashcode():
    state = random.getstate()
    random.seed(None)
    hashcode = '%08x' % random.getrandbits(32) # random seed does not fixed, wierd...
    random.setstate(state)
    return hashcode

# extract tuples from dictionary
def extract(d, keys):
    return [(key, d[key]) for key in keys]

class LoadArg(argparse.Action):
    def __call__ (self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

# for bach experiments, but combined with argparse and put this into your main.py
# batch_exps (or bash_tools exps) is more general. However, it is very difficult for them to control specific behaviour,
# and they need to deal with messy multi-processes
def read_args(args_path, timeout=30):
    lock_dir = Path(args_path.parent, '.lock')
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_fn = Path(lock_dir, args_path.stem)
    lock_fn.touch(exist_ok=True)
    with filelock.FileLock(lock_fn).acquire(timeout=timeout):
        with open(args_path) as f:
            jobs = f.read().splitlines(True)
        while jobs:
            job = jobs[0].strip()
            if not job or job.startswith('#'):
                jobs = jobs[1:]
            else:
                break
        if jobs:
            # skip empty line and comments
            args = shlex.split(jobs[0])
            with open(args_path, 'w') as f:
                f.writelines(jobs[1:])
        else:
            args = None
    return args

def push_args(args_str, args_path, timeout=30):
    lock_dir = Path(args_path.parent, '.lock')
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_fn = Path(lock_dir, args_path.stem)
    lock_fn.touch(exist_ok=True) # disadvantages: this will not be cleaned up
    with filelock.FileLock(lock_fn).acquire(timeout=timeout):
        with open(args_path) as f:
            jobs = f.read().splitlines(True)
        jobs.insert(0, args_str + '\n')
        with open(args_path, 'w') as f:
            f.writelines(jobs)

# input parser, arguments
# output a dictionary seperate args into different group
def group_args(parser, args):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    return args, arg_groups

def index_dict(l):
    l = list(enumerate(set(l)))
    i2e = dict(l)
    e2i = dict([(e, i) for i, e in l])
    return i2e, e2i

# for generator (function defined) to be able to get the current value
class cur_generator:
    def __init__(self, generator):
        self.__gen = generator

    def __iter__(self):
        return self

    def __next__(self):
        self.cur = next(self.__gen)
        return self.cur

    def __call__(self, *args, **kwargs): # don't call it twice!
        self.__gen = self.__gen(*args, **kwargs)
        return self

def with_cur(f):
    def g(*args, **kwargs):
        return cur_generator(f(*args, **kwargs))
    return g

@with_cur
def linspace(start, end, n, repeat_end=False):
    assert n > 1 
    cur = start
    for _ in range(n):
        yield cur 
        cur += (end - start) / (n - 1)
    if repeat_end:
        while True:
            yield cur 

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

def get_git_sha():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha

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
   
def run_supervised_steps(agent):
    config = agent.config
    t0 = time.time()
    while True:
        if config.log_interval and not agent.total_steps % config.log_interval and agent.total_steps:
            config.logger.info('total steps {}, NLL: {}, {:.2f} steps/s'.format( 
                agent.total_steps,
                agent.loss,
                config.log_interval / (time.time() - t0)))
            t0 = time.time()
            if config.eval_interval and not agent.total_steps / config.log_interval % config.eval_interval:
                acc = agent.eval_episodes()
                if config.save_interval and not agent.total_steps / config.log_interval / config.eval_interval % config.save_interval:
                    weight_dict = dict(
                        network=agent.network.state_dict(),
                        action_predictor=config.action_predictor.state_dict() if hasattr(config, 'action_predictor') else None,
                    )
                    config.logger.save_model('step-{}-acc-{:.2f}'.format(agent.total_steps, acc), weight_dict)
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()

def get_states_infos(env, discount):
    states, infos = [], []
    for index in env.unwrapped.train_combos:
        _states, _infos = env.last.get_teleportable_states(discount, index=index)
        states += _states
        infos += _infos 
    return states, infos

def save_abs(agent):
    config = agent.config
    env = config.eval_env.env.envs[0]
    all_states, all_infos = get_states_infos(env, config.discount)
    all_states = tensor(config.state_normalizer(all_states))
    all_infos = stack_dict(all_infos) 
    if agent.network.abs_encoder.abstract_type == 'sample':
        indices = agent.network.abs_encoder.get_indices(all_states, all_infos).detach().cpu().numpy()
        indices = [tuple(index) for index in indices]
        i2e, e2i = index_dict(indices)
        indices = [e2i[index] for index in indices]
    else:
        indices = agent.network.abs_encoder.get_indices(all_states, all_infos).detach().cpu().numpy()
    abs_map = {pos: i for pos, i in zip(all_infos['pos'], indices)}
    dir_path = Path(config.abs_save_path.parent, 'gen_abs_map')
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(Path(dir_path, '{}.pkl'.format(config.abs_save_path.name)), 'wb') as f:
        dill.dump(abs_map, f)

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
            if config.eval_interval and not agent.total_steps / config.log_interval % config.eval_interval:
                mean_return = agent.eval_episodes()
                if config.save_interval and not agent.total_steps / config.log_interval / config.eval_interval % config.save_interval:
                    weight_dict = dict(
                        network=agent.network.state_dict(),
                        action_predictor=config.action_predictor.state_dict() if hasattr(config, 'action_predictor') else None,
                    )
                    config.logger.save_model('step-{}-mean-{:.2f}'.format(stats['steps'], mean_return), weight_dict)
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

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

def fsave(data, fn, ftype):
    dirname = os.path.dirname(fn)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if ftype == 'json':
        with open(fn, 'w') as f:
            json.dump(data, f)
    elif ftype == 'pkl':
        with open(fn, 'wb') as f:
            dill.dump(data, f)        
    elif ftype == 'png':
        Image.fromarray(data).save(fn)
    else:
        raise Exception('unsupported file type: {}'.format(ftype))
 
