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
import datetime
import filelock
import torch
import time
import dill
import shlex
import shutil
from PIL import Image
from .torch_utils import *
try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path

def line_prepend(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content) # \r: return, \n: newline

# for bach experiments, but combined with argparse and put this into your main.py
# batch_exps (or bash_tools exps) is more general. However, it is very difficult for them to control specific behaviour,
# and they need to deal with messy multi-processes
def read_args(args_fn, timeout=30):
    lock_fn = Path(args_fn).with_name('.{}.lock'.format(Path(args_fn).stem))
    lock_fn.touch(exist_ok=True)
    with filelock.FileLock(lock_fn).acquire(timeout=timeout):
        with open(args_fn) as f:
            jobs = f.read().splitlines(True)
        while True:
            job = jobs[0].strip()
            if jobs and (not job or job.startswith('#')):
                jobs = jobs[1:]
            else:
                break
        if jobs:
            # skip empty line and comments
            args = shlex.split(jobs[0])
            with open(args_fn, 'w') as f:
                f.writelines(jobs[1:])
        else:
            args = None
    return args

def push_args(args_str, args_fn, timeout=30):
    lock_fn = Path(args_fn).with_name('.{}.lock'.format(Path(args_fn).stem))
    lock_fn.touch(exist_ok=True) # disadvantages: this will not be cleaned up
    with filelock.FileLock(lock_fn).acquire(timeout=timeout):
        with open(args_fn) as f:
            jobs = f.read().splitlines(True)
        jobs.insert(0, args_str + '\n')
        with open(args_fn, 'w') as f:
            f.writelines(jobs)

def index_dict(l):
    l = list(enumerate(set(l)))
    i2e = dict(l)
    e2i = dict([(e, i) for i, e in l])
    return i2e, e2i

# for generator (function defined) to be able to get the current value
class with_cur:
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
    save_dir = Path('data', 'models', config.log_name)
    if save_dir.exists(): shutil.rmtree(save_dir)
    mkdir(save_dir)
    t0 = time.time()
    while True:
        if config.log_interval and not agent.total_steps % config.log_interval and agent.total_steps:
            config.logger.info('total steps {}, NLL: {}, {:.2f} steps/s'.format( 
                agent.total_steps,
                agent.loss,
                config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            acc = agent.eval_episodes()
            if config.save_interval * config.eval_interval and not agent.total_steps % (config.save_interval * config.eval_interval):
                weight_dict = dict(
                    network=agent.network.state_dict(),
                    action_predictor=config.action_predictor.state_dict() if hasattr(config, 'action_predictor') else None,
                )
                torch.save(weight_dict, Path(save_dir, 'step-{}-acc-{:.2f}'.format(agent.total_steps, acc)))
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()

def run_steps(agent):
    config = agent.config
    save_dir = Path('data', 'models', config.log_name)
    if save_dir.exists(): shutil.rmtree(save_dir)
    mkdir(save_dir)
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
                acc = agent.eval_episodes()
                weight_dict = dict(
                    network=agent.network.state_dict(),
                    action_predictor=config.action_predictor.state_dict() if hasattr(config, 'action_predictor') else None,
                )
                if config.save_interval and not agent.total_steps / config.log_interval / config.eval_interval % config.save_interval:
                    torch.save(weight_dict, Path(save_dir, 'step-{}-mean-{:.2f}'.format(stats['steps'], acc)))
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()

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
 
