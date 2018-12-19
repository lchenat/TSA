#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
from ipdb import slaunch_ipdb_on_exception
import argparse

def _command_line_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--tag', type=str, required=True)
  parser.add_argument('--agent', type=str, default='tsa', choices=['tsa', 'baseline'])
  parser.add_argument('--num_abs', type=int, default=50)

  return parser


### tsa ###

def ppo_pixel_tsa(args):
    env_config = dict(
        map_names = ['map49'],
        train_combos = [(0, 1, 1)], # single task
        test_combos = [(0, 2, 2)],
        min_dis=10,
    )
    config = Config()
    config.log_name = '{}-{}'.format(ppo_pixel_tsa.__name__, args.tag)
    log_dir = get_log_dir(config.log_name)
    config.task_fn = lambda: GridWorldTask(env_config, log_dir=log_dir, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    config.num_workers = 8
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.99, eps=1e-5)

    config.state_dim = 512
    config.n_abs = args.num_abs # number of abstract state
    phi = AbstractedStateEncoder(config.n_abs, config.state_dim, TSAMiniConvBody(feature_dim=config.state_dim))
    actor = AbstractedActor(config.state_dim, config.action_dim) # given index, output distribution, hence embedding
    critic = VanillaNet(1, TSAMiniConvBody(feature_dim=config.state_dim))
    config.network_fn = lambda: CategoricalTSAActorCriticNet(config.action_dim, phi, actor, critic)
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.max_steps = int(2e7)
    config.save_interval = 0 # how many steps to save a model
    config.logger = get_logger(tag=config.log_name)
    run_steps(PPOAgent(config))

def ppo_pixel_baseline(args):
    env_config = dict(
        map_names = ['map49'],
        train_combos = [(0, 1, 1)], # single task
        test_combos = [(0, 2, 2)],
        min_dis=10,
    )
    config = Config()
    config.log_name = '{}-{}'.format(ppo_pixel_tsa.__name__, args.tag)
    log_dir = get_log_dir(config.log_name)
    config.task_fn = lambda: GridWorldTask(env_config, log_dir=log_dir, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    config.num_workers = 8

    config.state_dim = 512
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, TSAMiniConvBody())

    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.max_steps = int(2e7)
    config.save_interval = 0 # how many steps to save a model
    config.logger = get_logger(tag=config.log_name)
    run_steps(PPOAgent(config))

### end of tsa ###

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed(0)
    select_device(0)

    parser = _command_line_parser()
    args = parser.parse_args()

    with slaunch_ipdb_on_exception():
      if args.agent == 'tsa':
        ppo_pixel_tsa(args)
      elif args.agent == 'baseline':
        ppo_pixel_baseline(args)
      else:
        raise ValueError()

    # select_device(0)
