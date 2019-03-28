# used to generate experiments
from pathlib import Path
from deep_rl.utils.misc import cmd, cmd_run

def dump_args(f, args=None, kwargs=None):
    cmds = ''
    if args:
        cmds += ' '.join(args)
    if kwargs:
        cmds += ' '.join(['{} {}'.format(k, v) for k, v in kwargs.items()])
    f.write(cmds + '\n')

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
        for feat_dim in [50, 5]:
            if feat_dim != 512:
                kwargs['--feat_dim'] = feat_dim
                kwargs['--tag'] = feat_dim
            for g in range(9):
                kwargs['--env_config'] = 'data/env_configs/pick/nineroom/nineroom.{}'.format(g)
                for seed in range(5):
                    kwargs['--seed'] = seed
                    dump_args(f, kwargs=kwargs)


@cmd()
def train_reacher_cont():
    exp_path = Path('exps/reacher/train_cont')
    kwargs = {
        '--env': 'reacher',
        '--agent': 'fc_discrete',
        '--net': 'gaussian',
        '--hidden': 32,
        '--save_interval': 1,
        '--steps': 720000,
    }
    for feat_dim in [4, 8, 16, 32]:
        kwargs['--feat_dim'] = feat_dim
        kwargs['--tag'] = 

if __name__ == "__main__":
    cmd_run()
