#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from tensorboardX import SummaryWriter
import os
import glob
import numpy as np
import torch
import shutil
import random
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
from .misc import *
from pathlib import Path

base_log_dir = './log'

def get_logger(name, tags=None, skip=False, replace=False, level=logging.INFO):
    if tags['others'] is not None:
        log_format = Path(base_log_dir, tags['task'], tags['algo'], tags['others'], str(tags['seed']))
    else:
        log_format = Path(base_log_dir, tags['task'], tags['algo'], '_', str(tags['seed']))
    log_dir = Path('{}.{}'.format(log_format, get_time_str()))
    if not skip and log_exist(log_format):
        if not replace and stdin_choices('log exists, want to replace?', ['y', 'n']) == 'n':
            raise Exception('Error: log directory exists')
        remove_log(log_format)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not skip:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir, 'log.txt')
        log_path.touch()
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(level)
        logger.addHandler(fh)
    return Logger(logger, log_dir, skip)

def log_exist(log_format):
    files = glob.glob('{}.*-*'.format(log_format))
    files = [f for f in files if os.path.isdir(f)]
    return len(files)

def remove_log(log_format):
    for filename in glob.glob('{}.*-*'.format(log_format)):
        if os.path.isdir(filename):
            shutil.rmtree(filename)

def convert2str(*args):
    res = ''
    for obj in args:
        if isinstance(obj, dict):
            for k, v in obj.items():
                res += '{}: {}\n\n'.format(k, v)
        else:
            res += '{}\n\n'.format(obj)
    return res

class Logger(object):
    def __init__(self, vanilla_logger, log_dir, skip=False):
        self.log_dir = log_dir
        if not skip:
            #remove_tf_log(self.log_dir)
            self.writer = SummaryWriter(self.log_dir)
        if vanilla_logger is not None:
            self.info = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning
        self.skip = skip
        self.all_steps = {}

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag, value, step=None):
        if self.skip:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None):
        if self.skip:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)

    def add_text(self, tag, values):
        if self.skip:
            return
        self.writer.add_text(tag, convert2str(*values))

    def add_file(self, tag, value, step=None, ftype='pkl'):
        if self.skip:
            return
        if step is None:
            step = self.get_step(tag)
        fsave(value, Path(self.log_dir, tag, '{}.{}'.format(step, ftype)), ftype=ftype)

    def save_model(self, tag, value):
        if self.skip:
            return
        save_dir = Path(self.log_dir, 'models')
        mkdir(save_dir)
        torch.save(value, Path(save_dir, tag))
