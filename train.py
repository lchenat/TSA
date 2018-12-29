#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl.agent import *
from deep_rl.component import *
from deep_rl.network import *
from deep_rl.utils import *

from ipdb import slaunch_ipdb_on_exception
from termcolor import colored
import argparse
import dill

def _command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str, default='tsa', choices=['tsa', 'baseline'])
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--num_abs', type=int, default=50)
    parser.add_argument('-d', action='store_true')
    parser.add_argument('--abs_fn', type=str, default=None)

    return parser

def ppo_pixel_tsa(args):
    env_config = dict(
        map_names = ['map49'],
        train_combos = [(0, 1, 1)],
        #train_combos=[(0, 1, 1), (0, 2, 2), (0, 1, 2)],
        #train_combos=[(0, 1, 1), (0, 1, 7), (0, 1, 12)],
        test_combos = [(0, 2, 2)],
        min_dis=10,
    )
    config = Config()
    config.abs_dim = 512
    config.n_abs = 256 # number of abstract state, try large
    config.log_name = '{}-{}-n_abs-{}'.format(ppo_pixel_tsa.__name__, args.tag, config.n_abs)
    log_dir = get_log_dir(config.log_name)
    config.task_fn = lambda: GridWorldTask(env_config, log_dir=log_dir, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 8
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.99, eps=1e-5)
    #config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, TSAMiniConvBody())
    visual_body = TSAConvBody() # TSAMiniConvBody()
    ### VQ ###
    #abs_encoder = VQAbstractEncoder(config.n_abs, config.abs_dim, visual_body, abstract_type='max')
    #actor = LinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
    ### Prob ###
    #abs_encoder = ProbAbstractEncoder(config.n_abs, visual_body)
    #actor = EmbeddingActorNet(config.n_abs, config.action_dim, config.eval_env.n_tasks)
    ### Pos ###
    assert hasattr(args, 'abs_fn'), 'need args.abs_fn'
    with open(os.path.join('abs', '{}.pkl'.format(args.abs_fn)), 'rb') as f:
        abs_dict = dill.load(f)
        n_abs = len(set(abs_dict[0].values())) # only have 1 map!
    abs_encoder = PosAbstractEncoder(n_abs, abs_dict)
    actor = EmbeddingActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
    ##########
    critic = TSACriticNet(visual_body, config.eval_env.n_tasks)
    network = TSANet(config.action_dim, abs_encoder, actor, critic)
    config.network_fn = lambda: network
    ### aux loss ###
    config.action_predictor = ActionPredictor(config.action_dim, visual_body)
    ##########
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
    config.max_steps = 1e4 if args.d else int(2e7)
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
    config.log_name = '{}-{}'.format(ppo_pixel_baseline.__name__, args.tag)
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

if __name__ == '__main__':
    parser = _command_line_parser()
    args = parser.parse_args()
    if not args.d and is_git_diff():
        print(colored('please commit your changes before running new experiments!', 'red', attrs=['bold']))
        exit()

    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed(0)
    select_device(0)

    if args.d:
        context = slaunch_ipdb_on_exception
    else:
        context = with_null
    with context():
        if args.agent == 'tsa':
            ppo_pixel_tsa(args)
        elif args.agent == 'baseline':
            ppo_pixel_baseline(args)

