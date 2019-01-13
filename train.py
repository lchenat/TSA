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
    parser.add_argument('agent', default='tsa', choices=['tsa', 'baseline', 'supervised'])
    parser.add_argument('--net', default='prob', choices=['prob', 'vq', 'pos', 'kv'])
    parser.add_argument('--n_abs', type=int, default=512)
    parser.add_argument('--abs_fn', type=str, default=None)
    parser.add_argument('--env_config', type=str, default='data/env_configs/map49-single')
    parser.add_argument('--opt', choices=['vanilla', 'alt', 'diff'], default='vanilla')
    parser.add_argument('--opt_gap', nargs=2, type=int, default=[9, 9])
    parser.add_argument('--critic', default='visual', choices=['critic', 'abs'])
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--actor', choices=['linear', 'nonlinear'], default='nonlinear')
    parser.add_argument('--temperature', type=float, nargs='+', default=[1.0])
    parser.add_argument('-lr', nargs='+', type=float, default=[0.00025])
    parser.add_argument('-d', action='store_true')
    return parser

def set_optimizer_fn(args, config):
    if len(args.lr) < 2: args.lr.append(args.lr[0])
    if args.opt == 'vanilla':
        config.optimizer_fn = \
            lambda model: VanillaOptimizer(model.parameters(), torch.optim.RMSprop(model.parameters(), lr=args.lr[0], alpha=0.99, eps=1e-5), config.gradient_clip)
    elif args.opt == 'alt':
        def optimizer_fn(model):
            abs_params = list(model.abs_encoder.parameters())
            abs_ids = set(id(param) for param in abs_params)
            actor_params = list(model.actor.parameters()) + [param for param in model.critic.parameters() if id(param) not in abs_ids]
            abs_opt = torch.optim.RMSprop(abs_params, lr=args.lr[0], alpha=0.99, eps=1e-5)
            actor_opt = torch.optim.RMSprop(actor_params, lr=args.lr[1], alpha=0.99, eps=1e-5)
            return AlternateOptimizer([abs_params, actor_params], [abs_opt, actor_opt], args.opt_gap, config.gradient_clip)
        config.optimizer_fn = optimizer_fn
    elif args.opt == 'diff':
        def optimizer_fn(model):
            abs_params = list(model.abs_encoder.parameters())
            abs_ids = set(id(param) for param in abs_params)
            actor_params = list(model.actor.parameters()) + [param for param in model.critic.parameters() if id(param) not in abs_ids]
            return VanillaOptimizer(model.parameters(), 
                torch.optim.RMSprop(
                    [{'params': abs_params, 'lr': args.lr[0]}, 
                     {'params': actor_params, 'lr': args.lr[1]}], 
                    lr=args.lr[0],
                    alpha=0.99, 
                    eps=1e-5,
                ), 
                config.gradient_clip,
            )
        config.optimizer_fn = optimizer_fn
    else:
        raise Exception('unsupported optimizer type')

def set_network_fn(args, config):
    visual_body = TSAConvBody(3*config.env_config['window']) # TSAMiniConvBody()
    if args.net == 'vq':
        config.n_abs = args.n_abs
        config.log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), config.n_abs)
        abs_encoder = VQAbstractEncoder(config.n_abs, config.abs_dim, visual_body)
        if args.actor == 'nonlinear':
            actor = NonLinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
        else:
            actor = LinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
    elif args.net == 'prob':
        config.n_abs = args.n_abs
        config.log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), config.n_abs)
        if len(args.temperature) == 1:
            args.temperature = linspace(args.temperature[0], args.temperature[0], 2, repeat_end=True)
        elif len(args.temperature) == 3:
            args.temperature[2] = int(args.temperature[2])
            args.temperature = linspace(*args.temperature, repeat_end=True)
        else:
            raise Exception('this length is not gonna work')
        abs_encoder = ProbAbstractEncoder(config.n_abs, visual_body, temperature=args.temperature)
        actor = EmbeddingActorNet(config.n_abs, config.action_dim, config.eval_env.n_tasks)
    elif args.net == 'pos':
        assert args.abs_fn is not None, 'need args.abs_fn'
        with open(args.abs_fn, 'rb') as f:
            abs_dict = dill.load(f)
            n_abs = len(set(abs_dict[0].values())) # only have 1 map!
        config.n_abs = n_abs
        config.log_name = '{}-{}-{}-{}'.format(args.agent, args.net, lastname(args.env_config), lastname(args.abs_fn)[:-4])
        print(abs_dict)
        abs_encoder = PosAbstractEncoder(n_abs, abs_dict)
        actor = EmbeddingActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
    elif args.net == 'kv': # key-value
        config.n_abs = args.n_abs
        config.log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), config.n_abs)
        abs_encoder = KVAbstractEncoder(config.n_abs, config.abs_dim, visual_body)
        if args.actor == 'nonlinear':
            actor = NonLinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
        else:
            actor = LinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
    if args.critic == 'visual':
        critic_body = visual_body
    elif args.critic == 'abs':
        critic_body = abs_encoder
    critic = TSACriticNet(critic_body, config.eval_env.n_tasks)
    network = TSANet(config.action_dim, abs_encoder, actor, critic)
    config.network_fn = lambda: network
    ### aux loss ###
    config.action_predictor = ActionPredictor(config.action_dim, visual_body)
    ##########

def ppo_pixel_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        config.env_config = env_config
    config.abs_dim = 512
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 8
    set_network_fn(args, config)
    set_optimizer_fn(args, config)
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
    if args.tag: config.log_name += '-{}'.format(args.tag)
    config.logger = get_logger(tag=config.log_name)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    config.logger.save_file(env_config, 'env_config')
    run_steps(PPOAgent(config))

def ppo_pixel_baseline(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        n_tasks = len(env_config['train_combos'] + env_config['test_combos'])
        config.env_config = env_config
    config.log_name = '{}-{}-{}'.format(args.agent, args.net, lastname(args.env_config))
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    config.num_workers = 8
    config.state_dim = 512
    if args.opt == 'vanilla':
        config.optimizer_fn = \
            lambda model: VanillaOptimizer(model.parameters(), torch.optim.RMSprop(model.parameters(), lr=0.00025, alpha=0.99, eps=1e-5), config.gradient_clip)
    else:
        raise Exception('unsupported optimizer type')
    config.network_fn = lambda: CategoricalActorCriticNet(n_tasks, config.state_dim, config.action_dim, TSAMiniConvBody(3*env_config['window']))
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
    if args.tag: config.log_name += '-{}'.format(args.tag)
    config.logger = get_logger(tag=config.log_name)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    config.logger.save_file(env_config, 'env_config')
    run_steps(PPOAgent(config))

def supervised_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        n_tasks = len(env_config['train_combos'] + env_config['test_combos'])
        config.env_config = env_config
    config.log_name = '{}-{}-{}'.format(args.agent, args.net, lastname(args.env_config))
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    if args.opt == 'vanilla':
        config.optimizer_fn = lambda model: VanillaOptimizer(
            model.parameters(),
            #torch.optim.RMSprop(model.parameters(), lr=args.lr[0], alpha=0.99, eps=1e-5), 
            torch.optim.Adam(model.parameters(), lr=args.lr[0]),
            config.gradient_clip,
        )
    else:
        raise Exception('unsupported optimizer type')
    #set_network_fn(args, config)
    config.network_fn = lambda: CategoricalActorCriticNet(n_tasks, config.state_dim, config.action_dim, TSAMiniConvBody(3*env_config['window'])) # debug
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.log_interval = 1
    config.max_steps = 120 if args.d else int(10000)
    config.save_interval = 0 # how many steps to save a model
    config.eval_interval = 100
    if args.tag: config.log_name += '-{}'.format(args.tag)
    config.logger = get_logger(tag=config.log_name)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    config.logger.save_file(env_config, 'env_config')
    run_supervised_steps(SupervisedAgent(config))

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
        elif args.agent == 'supervised':
            supervised_tsa(args)

