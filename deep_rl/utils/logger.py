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
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
from .misc import *


base_log_dir = './tf_log'

def get_logger(tag=None, skip=False, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    if tag is not None:
        fh = logging.FileHandler('./log/%s.txt' % (tag,))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(level)
        logger.addHandler(fh)
    return Logger(logger, tag, skip)

def remove_tf_log(log_dir_tag):
    for filename in glob.glob(os.path.join(base_log_dir, '{}*'.format(log_dir_tag))):
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
    def __init__(self, vanilla_logger, log_dir_tag, skip=False):
        self.log_dir_tag = log_dir_tag
        self.log_dir = os.path.join(base_log_dir, '{}-{}'.format(log_dir_tag, get_time_str()))
        if not skip:
            remove_tf_log(log_dir_tag)
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
        self.writer.add_scalar(os.path.join(self.log_dir_tag, tag), value, step)

    def add_histogram(self, tag, values, step=None):
        if self.skip:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(os.path.join(self.log_dir_tag, tag), values, step)

    def add_text(self, tag, values):
        self.writer.add_text(os.path.join(self.log_dir_tag, tag), convert2str(*values))

    def add_file(self, tag, value, step=None, ftype='pkl'):
        if step is None:
            step = self.get_step(tag)
        fsave(value, os.path.join(self.log_dir, tag, '{}.{}'.format(step, ftype)), ftype=ftype)

    def save_file(self, data, fn, ftype='pkl'):
        fsave(data, os.path.join(self.log_dir, fn), ftype=ftype)
