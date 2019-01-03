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
    parser.add_argument('agent', default='tsa', choices=['tsa', 'baseline'])
    parser.add_argument('--net', default='prob', choices=['prob', 'vq', 'pos'])
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--n_abs', type=int, default=512)
    parser.add_argument('-d', action='store_true')
    parser.add_argument('--abs_fn', type=str, default=None)
    parser.add_argument('--env_config', type=str, default='data/env_configs/map49-single')
    parser.add_argument('--opt', choices=['vanilla', 'alt'], default='vanilla')
    return parser

def ppo_pixel_tsa(args):
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
    config = Config()
    config.abs_dim = 512
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 8
    visual_body = TSAConvBody() # TSAMiniConvBody()
    if args.net == 'vq':
        config.n_abs = args.n_abs
        config.log_name = '{}-{}-{}-n_abs-{}-{}'.format(args.agent, args.net, lastname(args.env_config), config.n_abs, args.tag)
        abs_encoder = VQAbstractEncoder(config.n_abs, config.abs_dim, visual_body, abstract_type='max')
        actor = LinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
    elif args.net == 'prob':
        config.n_abs = args.n_abs
        config.log_name = '{}-{}-{}-n_abs-{}-{}'.format(args.agent, args.net, lastname(args.env_config), config.n_abs, args.tag)
        abs_encoder = ProbAbstractEncoder(config.n_abs, visual_body)
        actor = EmbeddingActorNet(config.n_abs, config.action_dim, config.eval_env.n_tasks)
    elif args.net == 'pos':
        assert args.abs_fn is not None, 'need args.abs_fn'
        with open(args.abs_fn, 'rb') as f:
            abs_dict = dill.load(f)
            n_abs = len(set(abs_dict[0].values())) # only have 1 map!
        config.n_abs = n_abs
        config.log_name = '{}-{}-{}-{}-{}'.format(args.agent, args.net, lastname(args.env_config), lastname(args.abs_fn)[:-4], args.tag)
        print(abs_dict)
        abs_encoder = PosAbstractEncoder(n_abs, abs_dict)
        actor = EmbeddingActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
    critic = TSACriticNet(visual_body, config.eval_env.n_tasks)
    network = TSANet(config.action_dim, abs_encoder, actor, critic)
    config.network_fn = lambda: network
    ### aux loss ###
    config.action_predictor = ActionPredictor(config.action_dim, visual_body)
    ##########
    if args.opt == 'vanilla':
        config.optimizer_fn = \
            lambda model: VanillaOptimizer(model.parameters(), torch.optim.RMSprop(params, lr=0.00025, alpha=0.99, eps=1e-5), config.gradient_clip)
    else:
        def optimizer_fn(model):
            abs_params = list(model.abs_encoder.parameters()) + list(model.critic.parameters())
            actor_params = model.actor.parameters()
            abs_opt = torch.optim.RMSprop(abs_params, lr=0.00025, alpha=0.99, eps=1e-5)
            actor_opt = torch.optim.RMSprop(actor_params, lr=0.00025, alpha=0.99, eps=1e-5)
            return AlternateOptimizer([abs_params, actor_params], [abs_opt, actor_opt], [2, 4], config.gradient_clip)
        config.optimizer_fn = optimizer_fn
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
    config.max_steps = 1e4 if args.d else int(1.5e7)
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
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
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

