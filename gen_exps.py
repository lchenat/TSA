# used to generate experiments
import os
import random
import filelock
from pathlib import Path
from itertools import product
from deep_rl.utils.misc import cmd, cmd_run

random.seed(1)

@cmd('resume')
def resume_exp(exp_fn):
    exp_fn = Path(exp_fn)
    assert exp_fn.suffix == '.run', 'can only stop exp.run'
    lock_dir = Path(exp_fn.parent, '.lock')
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_fn = Path(lock_dir, exp_fn.stem)
    lock_fn.touch(exist_ok=True)
    with filelock.FileLock(lock_fn).acquire(timeout=30):
        with open(exp_fn) as f:
            lines = f.readlines()
            lines = [line[1:] for line in lines] # comment
        with open(exp_fn, 'w') as f:
            f.writelines(lines)

@cmd('stop')
def stop_exp(exp_fn):
    exp_fn = Path(exp_fn)
    assert exp_fn.suffix == '.run', 'can only stop exp.run'
    lock_dir = Path(exp_fn.parent, '.lock')
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_fn = Path(lock_dir, exp_fn.stem)
    lock_fn.touch(exist_ok=True)
    with filelock.FileLock(lock_fn).acquire(timeout=30):
        with open(exp_fn) as f:
            lines = f.readlines()
            lines = ['#' + line for line in lines] # comment
        with open(exp_fn, 'w') as f:
            f.writelines(lines)

def generate_cmd(args=None, kwargs=None):
    cmds = ''
    if args:
        cmds += ' '.join(args)
    cmds += ' '
    if kwargs:
        cmds += ' '.join(['{} {}'.format(k, v) for k, v in kwargs.items()])
    return cmds

def dump_args(f, args=None, kwargs=None):
    f.write(generate_cmd(args, kwargs) + '\n')

@cmd()
def train_nineroom():
    # variable: env_config, seed, (feat_dim, tag)
    exp_path = Path('exps/pick/nineroom/train')
    kwargs = {
        '--agent': 'tsa',
        '--visual': 'mini',
        '--net': 'baseline',
        '--gate': 'softplus',
        '--obs_type': 'mask',
        '--scale': 2,
        '--eval_interval': 15,
        '--save_interval': 1,
        '--steps': 500000,
    }
    # clean up the file
    open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        # 512, 20, 50 ,5
        for feat_dim in [512, 50, 20, 5]:
            if feat_dim != 512:
                kwargs['--feat_dim'] = feat_dim
                kwargs['--tag'] = feat_dim
            for g in range(8, 9):
                kwargs['--env_config'] = 'data/env_configs/pick/nineroom/nineroom.{}'.format(g)
                for seed in range(5):
                    kwargs['--seed'] = seed
                    dump_args(f, kwargs=kwargs)

@cmd()
def train_nineroom_relu():
    # variable: env_config, seed, (feat_dim, tag)
    exp_path = Path('exps/pick/nineroom/train_relu')
    kwargs = {
        '--agent': 'tsa',
        '--visual': 'mini',
        '--net': 'baseline',
        '--gate': 'relu',
        '--obs_type': 'mask',
        '--scale': 2,
        '--eval_interval': 15,
        '--save_interval': 1,
        '--steps': 500000,
    }
    # clean up the file
    open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        # 512, 20, 50 ,5
        for feat_dim in [512, 20]:
            kwargs['--feat_dim'] = feat_dim
            if feat_dim != 512:
                kwargs['--tag'] = '{}_relu'.format(feat_dim)
            else:
                kwargs['--tag'] = 'relu'.format(feat_dim)              
            for g in range(8, 9):
                kwargs['--env_config'] = 'data/env_configs/pick/nineroom/nineroom.{}'.format(g)
                for seed in range(5):
                    kwargs['--seed'] = seed
                    dump_args(f, kwargs=kwargs)

@cmd()
def train_reacher_cont():
    exp_path = Path('exps/reacher/train_cont')
    args = ['--no_goal']
    kwargs = {
        '--env': 'reacher',
        '--agent': 'fc_discrete',
        '--net': 'gaussian',
        '--save_interval': 1,
        '--steps': 720000,
    }
    open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for hidden in [4, 8, 16, 32]:
            kwargs['--hidden'] = hidden
            kwargs['--tag'] = hidden
            for g in range(1, 5):
                kwargs['--goal_fn'] = 'data/goals/reacher/4_corners/{}_corner'.format(g)
                for seed in range(5):
                    kwargs['--seed'] = seed
                    dump_args(f, args, kwargs)

@cmd()
def nineroom_nmf_search(base_dir, feat_dim, touch=True):
    exp_path = Path('exps/pick/nineroom/nmf_search_{}'.format(feat_dim))
    kwargs = { 
        '--agent': 'tsa',
        '--env_config': 'data/env_configs/pick/nineroom/nineroom.8',
        '--net': 'baseline',
        '--visual': 'mini',
        '--gate': 'softplus',
        '--feat_dim': int(feat_dim),
        '--load_part': 'abs',
        '--obs_type': 'mask',
        '--scale': 2,
        '--eval_interval': 15, 
        '--save_interval': 1,
        '--steps': 500000,
    }   
    if touch: open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for name in os.listdir(base_dir):
            for seed in range(3):
                kwargs['--weight'] = Path(base_dir, name)
                kwargs['--tag'] = 'from_nmf_{}-{}'.format(feat_dim, name.split('-')[1])
                kwargs['--seed'] = seed
                dump_args(f, kwargs=kwargs)

@cmd()
def nineroom_load_search(expname, base_dir, feat_dim):
    exp_path = Path('exps/pick/nineroom/{}'.format(expname))
    kwargs = { 
        '--agent': 'tsa',
        '--env_config': 'data/env_configs/pick/nineroom/nineroom.8',
        '--net': 'baseline',
        '--visual': 'mini',
        '--gate': 'softplus',
        '--feat_dim': int(feat_dim),
        '--load_part': 'abs',
        '--obs_type': 'mask',
        '--scale': 2,
        '--eval_interval': 15, 
        '--save_interval': 1,
        '--steps': 500000,
    }   
    open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for name in os.listdir(base_dir):
            for seed in range(3):
                kwargs['--weight'] = Path(base_dir, name)
                kwargs['--tag'] = '{}-{}-{}'.format(expname, feat_dim, name.split('-')[1])
                kwargs['--seed'] = seed
                dump_args(f, kwargs=kwargs)

@cmd()
def nineroom_kl_only_search(base_dir, feat_dim, touch=True):
    exp_path = Path('exps/pick/nineroom/kl_only_search_{}'.format(feat_dim))
    kwargs = { 
        '--agent': 'tsa',
        '--env_config': 'data/env_configs/pick/nineroom/nineroom.8',
        '--net': 'baseline',
        '--visual': 'mini',
        '--gate': 'softplus',
        '--feat_dim': int(feat_dim),
        '--load_part': 'abs',
        '--obs_type': 'mask',
        '--scale': 2,
        '--eval_interval': 15, 
        '--save_interval': 1,
        '--steps': 500000,
    }   
    if touch: open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for name in os.listdir(base_dir):
            for seed in range(3):
                kwargs['--weight'] = Path(base_dir, name)
                kwargs['--tag'] = 'kl_only_{}-{}'.format(feat_dim, name.split('-')[1])
                kwargs['--seed'] = seed
                dump_args(f, kwargs=kwargs)

@cmd()
def nineroom_finetune_search(feat_dim):
    exp_path = Path('exps/pick/nineroom/finetune_search_{}'.format(feat_dim))
    weights = {
        '5': 'log/pick.split.10-5/nmf_sample.baseline.n_abs-5/5/0.190330-184241/models/step-1049600-acc-2.28',
        '20': 'log/pick.split.10-20/nmf_sample.baseline.n_abs-20/20/0.190330-152420/models/step-1049600-acc-10.80',
        '50': 'log/pick.split.10-50/nmf_sample.baseline.n_abs-50/50/0.190330-184228/models/step-140800-acc-9.36',
    }
    kwargs = {
        '--agent': 'tsa',
        '--env_config': 'data/env_configs/pick/nineroom/nineroom.e8',
        '--net': 'baseline',
        '--visual': 'mini',
        '--gate': 'softplus',
        '--feat_dim': int(feat_dim),
        '--obs_type': 'mask',
        '--scale': 2,
        '--eval_interval': 15, 
        '--save_interval': 1,
        '--steps': 1200000,
        '--weight': weights[feat_dim],
    }
    cmds = []
    open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for lr in [0.000025, 0.00001, 0.000005]:
            for momentum in [0.1, 0.3, 0.5, 0.7, 0.9]: 
                for batch_size in [32, 64, 128]:
                    for rollout_length in [128, 256, 512]:
                        for num_workers in [16, 32]:
                            for seed in range(2):
                                kwargs['-lr'] = lr
                                kwargs['--momentum'] = momentum
                                kwargs['--batch_size'] = batch_size
                                kwargs['--rollout_length'] = rollout_length
                                kwargs['--seed'] = seed
                                kwargs['--num_workers'] = num_workers
                                kwargs['--tag'] = '{}-{}-{}-{}-{}-{}'.format(feat_dim, lr, momentum, batch_size, rollout_length, num_workers)
                                cmds.append(generate_cmd(kwargs=kwargs) + '\n')
                                #dump_args(f, kwargs=kwargs)
        random.shuffle(cmds)
        f.writelines(cmds)

@cmd()
def nineroom_actor_mimic_search(base_dir, feat_dim, touch=True):
    feat_dim = int(feat_dim)
    exp_path = Path('exps/pick/nineroom/actor_mimic_search_{}'.format(feat_dim))
    kwargs = { 
        '--agent': 'tsa',
        '--env_config': 'data/env_configs/pick/nineroom/nineroom.8',
        '--net': 'baseline',
        '--visual': 'mini',
        '--gate': 'softplus',
        '--feat_dim': feat_dim,
        '--load_part': 'abs',
        '--obs_type': 'mask',
        '--scale': 2,
        '--eval_interval': 15, 
        '--save_interval': 1,
        '--steps': 500000,
    }
    if touch: open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for name in os.listdir(base_dir):
            step = int(name.split('-')[1])
            if '--' in name: continue
            #if feat_dim == 50 and step <= 800000: continue
            #if feat_dim == 5 and step <= 1500000: continue
            for seed in range(3):
                kwargs['--weight'] = Path(base_dir, name)
                kwargs['--tag'] = 'actor_mimic_{}-{}'.format(feat_dim, step)
                kwargs['--seed'] = seed
                dump_args(f, kwargs=kwargs)

@cmd()
def nineroom_nmf_direct_search(feat_dim):
    feat_dim = int(feat_dim)
    exp_path = Path('exps/pick/nineroom/nmf_direct_search_{}'.format(feat_dim))
    kwargs = { 
        '--agent': 'nmf_direct',
        '--env_config': 'data/env_configs/pick/nineroom/nineroom.e8',
        '--net': 'baseline',
        '--visual': 'mini',
        '--gate': 'softplus',
        '--feat_dim': feat_dim,
        '--obs_type': 'mask',
        '--scale': 2,
        '--eval_interval': 15, 
        '--save_interval': 1,
        '--steps': 500000,
        '--expert': 'nineroom',
        '--expert_fn': 'data/experts/nineroom',
        '--seed': 0,
    }
    x_iters = [2, 4, 8, 16, 32]
    u_iters = [2, 4, 8, 16, 32]
    v_iters = [2, 4, 8]
    open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for x_iter, u_iter, v_iter in product(x_iters, u_iters, v_iters):
            kwargs['--x_iter'] = x_iter
            kwargs['--u_iter'] = u_iter
            kwargs['--v_iter'] = v_iter
            kwargs['--tag'] = 'nmf_direct_{}-{}-{}-{}'.format(feat_dim, x_iter, u_iter, v_iter)
            dump_args(f, kwargs=kwargs)

@cmd()
def nineroom_nmf_sample_abs():
    exp_path = Path('exps/pick/nineroom/nmf_sample_abs')
    kwargs = { 
        '--env': 'pick',
        '--env_config': 'data/env_configs/pick/nineroom/nineroom.e8',
        '--agent': 'nmf_sample',
        '--net': 'baseline',
        '--visual': 'mini',
        '--gate': 'softplus',
        '--scale': 2,
        '--obs_type': 'mask',
        '--feat_dim': 20,
        '--sample_fn': 'data/nmf_sample/pick/nineroom/split.10-20',
        '--save_interval': 1,
        '--abs_mean': 3,
    }
    open(exp_path, 'w').close()
    with open(exp_path, 'a+') as f:
        for abs_coeff in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0]:
            kwargs['--abs_coeff'] = abs_coeff
            kwargs['--tag'] = 'abs-{}'.format(abs_coeff)
            dump_args(f, kwargs=kwargs) 

if __name__ == "__main__":
    cmd_run()
