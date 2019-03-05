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
from collections import OrderedDict
from termcolor import colored
from pathlib import Path
import argparse
import dill
import copy
import socket
import traceback
import shutil

def _command_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('op', type=str, choices=['new', 'join'], # no default!
        help='create a new exp with name exp-tag or join an old one')
    parser.add_argument('exp', type=str, default='exps/exp', 
        help='path of the experiment file')
    parser.add_argument('--tag', type=str, default='0',
        help='suffix tag creating a new experiment')
    return parser

def _exp_parser():
    parser = argparse.ArgumentParser()
    algo = parser.add_argument_group('algo')
    algo.add_argument(
        '--agent', 
        default='tsa', 
        choices=[
            'tsa', 
            'baseline', 
            'supervised',
            'imitation',
            'transfer_a2c',
            'transfer_ppo',
            'transfer_distral',
            'PI',
            'fc_discrete', # use fc body, tabular cases
            'fc_continuous', # use fc body
            'nmf_sample', # non-tabular cases
        ],
    )
    # environment (the task setting, first level directory)
    task = parser.add_argument_group('task')
    task.add_argument('--env', default='pick', choices=['pick', 'reach', 'grid'])
    task.add_argument('-l', type=int, default=16)
    task.add_argument('-T', type=int, default=100)
    task.add_argument('--window', type=int, default=1)
    task.add_argument('--env_config', type=str, default='data/env_configs/pick/map49-n_goal-2-min_dis-4')
    ## simple_grid only
    task.add_argument('--map_name', type=str, default='fourroom')
    task.add_argument('--goal_loc', type=int, nargs=2, default=(9, 9))
    ##
    task.add_argument('--discount', type=float, default=0.99)
    task.add_argument('--min_dis', type=int, default=10)
    task.add_argument('--task_config', type=str, default=None) # read from file
    # network
    algo.add_argument('--visual', choices=['mini', 'normal', 'large'], default='mini')
    algo.add_argument('--net', default='prob', choices=['prob', 'vq', 'pos', 'sample', 'baseline', 'i2a', 'bernoulli', 'map', 'imap'])
    algo.add_argument('--n_abs', type=int, default=512)
    algo.add_argument('--abs_fn', type=str, default=None)
    algo.add_argument('--actor', choices=['linear', 'nonlinear'], default='nonlinear')
    algo.add_argument('--critic', default='visual', choices=['critic', 'abs'])
    algo.add_argument('--rate', type=float, default=1)
    algo.add_argument('--rollout_length', type=int, default=128) # works for PPO only
    ## simple grid only
    task.add_argument('--hidden', type=int, nargs='+', default=(16,))
    task.add_argument('--sample_fn', type=str, default=None) # only currently, it is actually general
    ##
    # transfer network
    algo.add_argument('--t_net', default='prob', choices=['prob', 'vq', 'pos', 'sample', 'baseline', 'i2a', 'bernoulli'])
    algo.add_argument('--t_n_abs', type=int, default=512)
    algo.add_argument('--t_abs_fn', type=str, default=None)
    algo.add_argument('--t_actor', choices=['linear', 'nonlinear'], default='nonlinear')
    algo.add_argument('--distill_w', type=float, default=0.1)
    algo.add_argument('--alpha', type=float, default=0.5)
    algo.add_argument('--beta', type=float, default=1.0)
    # network setting
    algo.add_argument('--label', choices=['action', 'abs'], default='action')
    algo.add_argument('--weight', type=str, default=None)
    algo.add_argument('--load_part', default='all', choices=['all', 'abs'])
    algo.add_argument('--fix_abs', action='store_true')
    algo.add_argument('--temperature', type=float, nargs='+', default=[1.0])
    # aux loss
    algo.add_argument('--pred_action', action='store_true')
    algo.add_argument('--recon', action='store_true')
    algo.add_argument('--trans', action='store_true')
    algo.add_argument('--reg_abs_fn', type=str, default=None)
    algo.add_argument('--reg_abs_weight', type=float, default=1.0)
    # optimization
    algo.add_argument('--opt', choices=['vanilla', 'alt', 'diff'], default='vanilla')
    algo.add_argument('--opt_gap', nargs=2, type=int, default=[9, 9])
    algo.add_argument('-lr', nargs='+', type=float, default=[0.00025])
    algo.add_argument('--algo_config', type=str, default=None) # read from file
    # others, should not affect performance (except seed)
    parser.add_argument('--mode', default='train', choices=['train', 'save_abs'])
    parser.add_argument('--save_interval', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--skip', action='store_true') # skip logging
    parser.add_argument('-d', action='store_true')
    return parser

def record_run(args_str):
    hash_code = get_hashcode()
    run_path = Path('log', 'run')
    run_path.touch()
    line_prepend(run_path, '{} {}: {}'.format(socket.gethostname(), hash_code, args_str))
    return hash_code

# self-defined parse function
def parse(parser, *args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for config in [getattr(args, attr) for attr in ['task_config', 'algo_config']]: # no constraint on the argument group
        if config is not None:
            with open(config) as f:
                parser.parse_known_args(f.read().split(), args)            
    return args, group_args(parser, args)

# self-defined logname function
# return task name, algo name and the rest
def get_log_tags(args):
    tags = dict()
    if args.task_config:
        tags['task'] = Path(args.task_config).stem
    else:
        tags['task'] = '.'.join([
            args.env,
            Path(args.env_config).name,
            'min_dis-{}'.format(args.min_dis),
        ])
    if args.algo_config:
        tags['algo'] = Path(args.algo_config).stem
    else:
        tags['algo'] = args.algo_name
    tags['others'] = args.tag
    tags['seed'] = args.seed
    return tags

def set_optimizer_fn(args, config):
    if len(args.lr) < 2: args.lr.append(args.lr[0])
    if args.opt == 'vanilla':
        def optimizer_fn(model):
            params = filter(lambda p: p.requires_grad, model.parameters())
            return VanillaOptimizer(params, torch.optim.RMSprop(params, lr=args.lr[0], alpha=0.99, eps=1e-5), config.gradient_clip)
        config.optimizer_fn = optimizer_fn
    elif args.opt == 'alt':
        def optimizer_fn(model):
            abs_params = list(model.abs_encoder.parameters())
            abs_ids = set(id(param) for param in abs_params)
            actor_params = list(model.actor.parameters()) + [param for param in model.critic.parameters() if id(param) not in abs_ids]
            # filter
            abs_params = filter(lambda p: p.requires_grad, abs_params)
            actor_params = filter(lambda p: p.requires_grad, actor_params)
            abs_opt = torch.optim.RMSprop(abs_params, lr=args.lr[0], alpha=0.99, eps=1e-5)
            actor_opt = torch.optim.RMSprop(actor_params, lr=args.lr[1], alpha=0.99, eps=1e-5)
            return AlternateOptimizer([abs_params, actor_params], [abs_opt, actor_opt], args.opt_gap, config.gradient_clip)
        config.optimizer_fn = optimizer_fn
    elif args.opt == 'diff':
        def optimizer_fn(model):
            abs_params = list(model.abs_encoder.parameters())
            abs_ids = set(id(param) for param in abs_params)
            actor_params = list(model.actor.parameters()) + [param for param in model.critic.parameters() if id(param) not in abs_ids]
            abs_params = filter(lambda p: p.requires_grad, abs_params)
            actor_params = filter(lambda p: p.requires_grad, actor_params)
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

def process_temperature(temperature):
    if len(temperature) == 1:
        temperature = linspace(temperature[0], temperature[0], 2, repeat_end=True)
    elif len(temperature) == 3:
        temperature[2] = int(temperature[2])
        temperature = linspace(*temperature, repeat_end=True)
    else:
        raise Exception('this length is not gonna work')
    return temperature

def get_visual_body(args, config):
    if args.visual == 'mini':
        visual_body = TSAMiniConvBody(3*config.env_config['main']['window'])
    elif args.visual == 'normal':
        visual_body = TSAConvBody(3*config.env_config['main']['window']) 
    elif args.visual == 'large':
        visual_body = LargeTSAMiniConvBody(3*config.env_config['main']['window'])
    if args.reg_abs_fn is not None:
        with open(args.reg_abs_fn, 'rb') as f:
            abs_dict = dill.load(f)
        config.reg_abs = RegAbs(visual_body, abs_dict, args.reg_abs_weight)
    return visual_body

def set_aux_network(visual_body, args, config):
    if args.pred_action:
        config.action_predictor = ActionPredictor(config.action_dim, visual_body)
    if args.recon:
        config.recon = UNetReconstructor(visual_body, 3*config.env_config['main']['window'])
    if args.trans:
        config.trans = TransitionModel(visual_body, config.action_dim, 3*config.env_config['main']['window'])

# deal with loading and fixing weight
def process_weight(network, args, config):
    if args.weight is not None:
        weight_dict = network.state_dict()
        # this is error prone, if I change structure of weight_dict, it does not give error
        if args.load_part == 'all':
            load_filter = lambda x: True
        else:
            if hasattr(network, 'abs_encoder'):
                load_filter = lambda x: x.startswith('abs_encoder')
            else:
                load_filter = lambda x: x.startswith('network.phi_body')
        loaded_weight_dict = {k: v for k, v in torch.load(
            args.weight,
            map_location=lambda storage, loc: storage)['network'].items()
            if ((k in weight_dict) and load_filter(k))}
        weight_dict.update(loaded_weight_dict)
        network.load_state_dict(weight_dict)
        if 'action_predictor' in weight_dict:
            config.action_predictor.load_state_dict(weight_dict['action_predictor'])
    if args.fix_abs:
        if hasattr(network, 'abs_encoder'):
            for p in network.abs_encoder.parameters():
                p.requires_grad = False
        else: # baseline
            for p in network.network.phi_body.parameters():
                p.requires_grad = False

# for normal tsa
def get_network(visual_body, args, config):
    if args.net == 'baseline':
        algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
        network = CategoricalActorCriticNet(
            config.eval_env.n_tasks,
            config.state_dim,
            config.action_dim, 
            visual_body,
            actor_body=FCBody(visual_body.feature_dim, (args.n_abs,)),
        )
    else:
        if args.net == 'vq':
            algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
            abs_encoder = VQAbstractEncoder(args.n_abs, config.abs_dim, visual_body)
            if args.actor == 'nonlinear':
                actor = NonLinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
            else:
                actor = LinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'prob':
            algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
            temperature = process_temperature(args.temperature)
            abs_encoder = ProbAbstractEncoder(args.n_abs, visual_body, temperature=temperature)
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'pos':
            assert args.abs_fn is not None, 'need args.abs_fn'
            with open(args.abs_fn, 'rb') as f:
                abs_dict = dill.load(f)
                n_abs = len(set(abs_dict[0].values())) # only have 1 map!
            algo_name = '.'.join([args.agent, args.net, Path(args.abs_fn).stem])
            print(abs_dict)
            abs_encoder = PosAbstractEncoder(n_abs, abs_dict)
            if args.actor == 'linear':
                actor = LinearActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
            elif args.actor == 'nonlinear':
                actor = NonLinearActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'sample':
            temperature = process_temperature(args.temperature)
            algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
            abs_encoder = SampleAbstractEncoder(args.n_abs, visual_body, temperature=temperature)
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'i2a':
            temperature = process_temperature(args.temperature)
            algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
            abs_encoder = I2AAbstractEncoder(args.n_abs, visual_body, temperature=temperature)
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'bernoulli':
            temperature = process_temperature(args.temperature)
            algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
            abs_encoder = BernoulliAbstractEncoder(args.n_abs, visual_body, temperature=temperature)
            if args.actor == 'linear':
                actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
            else:
                actor = NonLinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        else:
            raise Exception('unsupported network type')
        if args.critic == 'visual':
            critic_body = visual_body
        elif args.critic == 'abs':
            critic_body = abs_encoder
        critic = TSACriticNet(critic_body, config.eval_env.n_tasks)
        network = TSANet(config.action_dim, abs_encoder, actor, critic)
    return network, algo_name

def get_grid_network(args, config):
    n_tasks = config.eval_env.n_tasks
    algo_name = [args.agent, args.net]
    if args.net == 'baseline':
        network = CategoricalActorCriticNet(
            n_tasks,
            config.state_dim,
            config.action_dim,
            FCBody(
                config.state_dim, 
                hidden_units=tuple(args.hidden)
            ),
        )
        return network, '.'.join(algo_name)
    elif args.net == 'map':
        assert args.abs_fn is not None, 'need args.abs_fn'
        with open(args.abs_fn, 'rb') as f:
            abs_dict = dill.load(f)
            n_abs = len(list(abs_dict.values())[0]) # this is the length of feature vector
        #print(abs_dict)
        def abs_f(states):
            np_states = to_np(states)
            abs_s = np.array([abs_dict[tuple(s)] for s in np_states])
            return tensor(abs_s)
        abs_encoder = MapAbstractEncoder(n_abs, abs_f)
        if args.actor == 'linear':
            actor = LinearActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.actor == 'nonlinear':
            actor = NonLinearActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
        algo_name.append(Path(args.abs_fn).name)
    elif args.net == 'imap': # take argmax
        assert args.abs_fn is not None, 'need args.abs_fn'
        with open(args.abs_fn, 'rb') as f:
            abs_dict = dill.load(f)
            abs_dict = {k: v.argmax() for k, v in abs_dict.items()}
            n_abs = max(set(abs_dict.values())) + 1 # only have 1 map!, don't want to map it back again
        print(abs_dict)
        def abs_f(states):
            np_states = to_np(states)
            abs_s = np.array([abs_dict[tuple(s)] for s in np_states])
            return one_hot.encode(tensor(abs_s, dtype=torch.long), n_abs)
        abs_encoder = MapAbstractEncoder(n_abs, abs_f)
        if args.actor == 'linear':
            actor = LinearActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.actor == 'nonlinear':
            actor = NonLinearActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
        algo_name.append(Path(args.abs_fn).name)
    elif args.net == 'prob': # used to approximate NMF, we also need non-negative net
        assert args.sample_fn is not None, 'need args.sample_fn'
        fc_body = FCBody(config.state_dim, hidden_units=tuple(args.hidden))
        temperature = process_temperature(args.temperature)
        abs_encoder = ProbAbstractEncoder(args.n_abs, fc_body, temperature=temperature)
        if args.actor == 'linear':
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.actor == 'nonlinear':
            actor = NonLinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
    else:
        raise Exception('unsupport data type')
    critic_body = abs_encoder
    critic = TSACriticNet(critic_body, config.eval_env.n_tasks)
    network = TSANet(config.action_dim, abs_encoder, actor, critic)
    return network, '.'.join(algo_name)

def ppo_pixel_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config['min_dis'] = args.min_dis
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 8
    visual_body = get_visual_body(args, config)
    network, args.algo_name = get_network(visual_body, args, config)
    config.network_fn = lambda: network
    set_aux_network(visual_body, args, config)
    process_weight(network, args, config)
    set_optimizer_fn(args, config)
    config.state_normalizer = ImageNormalizer()
    config.discount = args.discount
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = args.rollout_length
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.max_steps = 1e4 if args.d else int(1.5e7)
    if args.steps is not None: config.max_steps = args.steps
    config.save_interval = args.save_interval
    if args.mode == 'train':
        config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
        config.logger.add_text('Configs', [{
            'git sha': get_git_sha(),
            **vars(args),
            }])
        run_steps(PPOAgent(config))
    else:
        config.abs_save_path = Path(args.weight)
        save_abs(PPOAgent(config))

def fc_discrete(args):
    env_config = dict(
        main=dict(
            map_name=args.map_name,
            #init_loc=(1, 1),
            goal_loc=tuple(args.goal_loc),
            min_dis=args.min_dis,
        ),
        T=args.T, # 250?
    )
    config = Config()
    config.num_workers = 5
    config.task_fn = lambda: DiscreteGridTask(env_config, num_envs=config.num_workers)
    config.eval_env = DiscreteGridTask(env_config)
    def optimizer_fn(model):
        params = filter(lambda p: p.requires_grad, model.parameters())
        return VanillaOptimizer(params, torch.optim.RMSprop(params, 0.001), config.gradient_clip)
    config.optimizer_fn = optimizer_fn
    #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    network, args.algo_name = get_grid_network(args, config)
    config.network_fn = lambda: network
    config.discount = args.discount
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.rollout_length = args.rollout_length
    config.optimization_epochs = 10
    config.mini_batch_size = 32 * 5
    config.ppo_ratio_clip = 0.2
    config.log_interval = args.rollout_length * config.num_workers * 10

    config.max_steps = 180000
    if args.steps is not None: config.max_steps = args.steps
    config.eval_interval = 5
    config.save_interval = args.save_interval
    log_tags = dict(
        task='.'.join([
            args.env,
            env_config['main']['map_name'],
            '{}_{}'.format(*env_config['main']['goal_loc']),
            'md-{}'.format(env_config['main']['min_dis']),
            'T-{}'.format(env_config['T']),
        ]),
        algo=args.algo_name,
        others=args.tag,
        seed=args.seed,
    )
    config.logger = get_logger(args.hash_code, tags=log_tags, skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    run_steps(PPOAgent(config))

def nmf_sample(args):
    config = Config()
    config.eval_env = 
    def optimizer_fn(model):
        params = filter(lambda p: p.requires_grad, model.parameters())
        return VanillaOptimizer(params, torch.optim.RMSprop(params, 0.001), config.gradient_clip)
    config.optimizer_fn = optimizer_fn
    assert args.sample_fn is not None, 'need args.sample_fn'
    with open(args.sample_fn, 'rb') as f:
        sample_dict = dill.load(f) # abs, policy
        args.n_abs = sample_dict['abs'].shape[1]
        n_tasks = len(sample_dict['policies'])
    network, _ = get_grid_network(args, config)
    args.algo_name = '.'.join(['nmf_sample'])
    config.sample_dict = sample_dict
    config.network_fn = lambda: network
    config.gradient_clip = 5
    config.batch_size = 32 * n_tasks
    config.log_interval = config.batch_size * 10

    config.max_steps = 180000
    if args.steps is not None: config.max_steps = args.steps
    config.eval_interval = 5
    config.save_interval = args.save_interval
    log_tags = dict(
        task=Path(args.sample_fn).name,
        algo=args.algo_name,
        others=args.tag,
        seed=args.seed,
    )
    config.logger = get_logger(args.hash_code, tags=log_tags, skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    run_steps(NMFAgent(config))

def ppo_pixel_baseline(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config['min_dis'] = args.min_dis
        n_tasks = len(env_config['train_combos'] + env_config['test_combos'])
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    args.algo_name = args.agent
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    config.num_workers = 8
    config.state_dim = 512
    if args.opt == 'vanilla':
        config.optimizer_fn = \
            lambda model: VanillaOptimizer(model.parameters(), torch.optim.RMSprop(model.parameters(), lr=0.00025, alpha=0.99, eps=1e-5), config.gradient_clip)
    else:
        raise Exception('unsupported optimizer type')
    visual_body = TSAMiniConvBody(3*env_config['main']['window'])
    config.network_fn = lambda: CategoricalActorCriticNet(n_tasks, config.state_dim, config.action_dim, visual_body)
    if args.recon:
        config.recon = UNetReconstructor(visual_body, 3*config.env_config['main']['window'])
    config.state_normalizer = ImageNormalizer()
    config.discount = args.discount
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = args.rollout_length
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.max_steps = 1e4 if args.d else int(1.5e7)
    if args.steps is not None: config.max_steps = args.steps
    config.save_interval = args.save_interval
    config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    run_steps(PPOAgent(config))

def supervised_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config['min_dis'] = args.min_dis
        n_tasks = len(env_config['train_combos'] + env_config['test_combos'])
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    if args.opt == 'vanilla':
        config.optimizer_fn = lambda model: VanillaOptimizer(
            model.parameters(),
            #torch.optim.RMSprop(model.parameters(), lr=args.lr[0], alpha=0.99, eps=1e-5), 
            torch.optim.Adam(model.parameters(), lr=args.lr[0]),
            config.gradient_clip,
        )
    else:
        raise Exception('unsupported optimizer type')
    config.label = args.label
    if args.label == 'abs': # for prob only
        assert args.abs_fn is not None, 'need args.abs_fn'
        with open(args.abs_fn, 'rb') as f:
            abs_dict = dill.load(f)
            n_abs = len(set(abs_dict[0].values())) # only have 1 map!
            print(abs_dict)
        args.n_abs = n_abs # important!
        config.abs_network_fn = lambda: PosAbstractEncoder(n_abs, abs_dict)
    visual_body = get_visual_body(args, config)
    network, args.algo_name = get_network(visual_body, args, config)
    args.algo_name += '.label-{}'.format(args.label)
    config.network_fn = lambda: network
    set_aux_network(visual_body, args, config)
    process_weight(network, args, config)
    #config.network_fn = lambda: CategoricalActorCriticNet(n_tasks, config.state_dim, config.action_dim, TSAMiniConvBody(3*env_config['window'])) # debug
    config.state_normalizer = ImageNormalizer()
    config.discount = args.discount
    config.log_interval = 1
    config.max_steps = 10000 if args.d else int(10000)
    if args.steps is not None: config.max_steps = args.steps
    config.save_interval = 1 # how many steps to save a model
    config.eval_interval = 100
    config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    run_supervised_steps(SupervisedAgent(config))

def imitation_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config['min_dis'] = args.min_dis
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 8
    visual_body = get_visual_body(args, config)
    network, args.algo_name = get_network(visual_body, args, config)
    config.network_fn = lambda: network
    set_aux_network(visual_body, args, config)
    process_weight(network, args, config)
    set_optimizer_fn(args, config)
    config.state_normalizer = ImageNormalizer()
    config.discount = args.discount
    config.gradient_clip = 0.5
    config.rollout_length = args.rollout_length
    config.log_interval = config.num_workers * config.rollout_length
    config.eval_interval = 100 # in terms of log interval
    config.max_steps = 3e7 if args.d else int(3e7)
    if args.steps is not None: config.max_steps = args.steps
    config.save_interval = 1 # in terms of eval interval
    config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    run_steps(ImitationAgent(config))

# TODO:
# you need to set log_name
def transfer_ppo_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config['min_dis'] = args.min_dis
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 8
    # source: the one with abstraction, try to process weight
    visual_body = get_visual_body(args, config)
    network, args.algo_name = get_network(visual_body, args, config)
    config.source_fn = lambda: network
    process_weight(network, args, config)
    # target: the one transfer to, aux for this
    t_args = copy.deepcopy(args)
    t_args.net = args.t_net
    t_args.n_abs = args.t_n_abs
    t_args.abs_fn = args.t_abs_fn
    t_args.actor = args.t_actor
    visual_body = get_visual_body(t_args, config)
    network, _ = get_network(visual_body, t_args, config)
    config.target_fn = lambda: network
    set_aux_network(visual_body, t_args, config)
    config.distill_w = args.distill_w
    #set_optimizer_fn(args, config)
    def optimizer_fn(source, target):
        params = filter(lambda p: p.requires_grad, list(source.parameters())+list(target.parameters()))
        return VanillaOptimizer(params, torch.optim.RMSprop(params, lr=args.lr[0], alpha=0.99, eps=1e-5), config.gradient_clip)
    config.optimizer_fn = optimizer_fn
    config.state_normalizer = ImageNormalizer()
    config.discount = args.discount
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = args.rollout_length
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.max_steps = 1.5e7 if args.d else int(1.5e7)
    if args.steps is not None: config.max_steps = args.steps
    config.save_interval = args.save_interval
    args.algo_name += '.w-{}'.format(config.distill_w)
    config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    run_steps(TransferPPOAgent(config))

def transfer_a2c_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config['min_dis'] = args.min_dis
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 16
    visual_body = get_visual_body(args, config)
    network, args.algo_name = get_network(visual_body, args, config)
    config.source_fn = lambda: network
    process_weight(network, args, config)
    # target: the one transfer to, aux for this
    t_args = copy.deepcopy(args)
    t_args.net = args.t_net
    t_args.n_abs = args.t_n_abs
    t_args.abs_fn = args.t_abs_fn
    t_args.actor = args.t_actor
    visual_body = get_visual_body(t_args, config)
    network, _ = get_network(visual_body, t_args, config)
    config.target_fn = lambda: network
    set_aux_network(visual_body, t_args, config)
    config.distill_w = args.distill_w
    def optimizer_fn(source, target):
        params = filter(lambda p: p.requires_grad, list(source.parameters())+list(target.parameters()))
        return VanillaOptimizer(params, torch.optim.RMSprop(params, lr=args.lr[0], alpha=0.99, eps=1e-5), config.gradient_clip)
    config.optimizer_fn = optimizer_fn
    config.state_normalizer = ImageNormalizer()
    config.discount = args.discount
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = 1.5e7 if args.d else int(1.5e7)
    if args.steps is not None: config.max_steps = args.steps
    config.save_interval = args.save_interval
    args.algo_name += '.w-{}'.format(config.distill_w)
    config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    run_steps(TransferA2CAgent(config))

def transfer_distral_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config['min_dis'] = args.min_dis
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 16
    visual_body = get_visual_body(args, config)
    network, args.algo_name = get_network(visual_body, args, config)
    config.source_fn = lambda: network
    process_weight(network, args, config)
    # target: the one transfer to, aux for this
    t_args = copy.deepcopy(args)
    t_args.net = args.t_net
    t_args.n_abs = args.t_n_abs
    t_args.abs_fn = args.t_abs_fn
    t_args.actor = args.t_actor
    visual_body = get_visual_body(t_args, config)
    network, _ = get_network(visual_body, t_args, config)
    config.target_fn = lambda: network
    set_aux_network(visual_body, t_args, config)
    config.distill_w = args.distill_w
    def optimizer_fn(source, target):
        params = filter(lambda p: p.requires_grad, list(source.parameters())+list(target.parameters()))
        return VanillaOptimizer(params, torch.optim.RMSprop(params, lr=args.lr[0], alpha=0.99, eps=1e-5), config.gradient_clip)
    config.optimizer_fn = optimizer_fn
    config.state_normalizer = ImageNormalizer()
    config.discount = args.discount
    config.use_gae = True
    config.gae_tau = 1.0
    #config.entropy_weight = 0.01
    config.alpha = args.alpha
    config.beta = args.beta
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = 1.5e7 if args.d else int(1.5e7)
    if args.steps is not None: config.max_steps = args.steps
    config.save_interval = args.save_interval
    args.algo_name += '-w-{}'.format(config.distill_w)
    config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    run_steps(TransferDistralAgent(config))

def ppo_pixel_PI(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config['min_dis'] = args.min_dis
        n_tasks = len(env_config['train_combos'] + env_config['test_combos'])
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    args.algo_name = '.'.join([args.agent, args.net])
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    config.num_workers = 8
    config.rollout_length = args.rollout_length
    #config.state_dim = 512
    with open(args.abs_fn, 'rb') as f:
        abs_dict = dill.load(f)
        n_abs = len(set(abs_dict[0].values())) # only have 1 map!
        config.n_abs = n_abs
    log_name = '{}-{}-{}-{}'.format(args.agent, args.net, lastname(args.env_config), lastname(args.abs_fn)[:-4])
    print(abs_dict)
    abs_encoder = PosAbstractEncoder(n_abs, abs_dict)
    actor = TabularActor(n_abs, config.action_dim, config.eval_env.n_tasks)
    network = TSANet(config.action_dim, abs_encoder, actor, critic=None)
    config.network_fn = lambda: network
    #config.state_normalizer = ImageNormalizer()
    config.rate = args.rate
    config.discount = args.discount
    config.log_interval = config.rollout_length * config.num_workers
    config.max_steps = int(5e5) if args.d else int(5e5)
    if args.steps is not None: config.max_steps = args.steps
    config.save_interval = args.save_interval
    config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    run_steps(PIAgent(config))

if __name__ == '__main__':
    command_args = _command_parser().parse_args()
    parser = _exp_parser()
    if command_args.op == 'new':
        assert Path(command_args.exp).suffix == '', 'can only create new experiment on file without suffix'
        exp_path = Path('{}-{}.run'.format(command_args.exp, command_args.tag))
        if not exp_path.exists() or stdin_choices('{} exists, want to replace?'.format(exp_path), ['y', 'n']):
            shutil.copy(command_args.exp, str(exp_path))
    else: # join
        exp_path = Path(command_args.exp)
        assert exp_path.suffix == '.run', 'only support run filetype, name: {}, suffix: {}'.format(exp_path, exp_path.suffix)
    while True:
        args = read_args(exp_path)
        if args is None: break
        args_str = ' '.join(args)
        # save a hash code and push it to a file, so that it is easy to keey track of which program we have running
        exp_finished = False
        try:
            print(args)
            args, arg_groups = parse(parser, args)
            args.hash_code = record_run(args_str)
            if not args.d and is_git_diff():
                print(colored('please commit your changes before running new experiments!', 'red', attrs=['bold']))
                break
            if args.env == 'pick':
                Task = PickGridWorldTask
            elif args.env == 'reach':
                Task = ReachGridWorldTask
            elif args.env == 'grid':
                Task = DiscreteGridTask

            mkdir('log')
            set_one_thread()
            random_seed(args.seed)
            select_device(-1 if args.cpu else 0)

            if args.d:
                context = slaunch_ipdb_on_exception
            else:
                context = with_null
            # quit ipdb will jump out of this, therefore should put exp_finished inside context
            with context(): 
                if args.agent == 'tsa':
                    ppo_pixel_tsa(args)
                elif args.agent == 'baseline':
                    ppo_pixel_baseline(args)
                elif args.agent == 'supervised':
                    supervised_tsa(args)
                elif args.agent == 'imitation':
                    imitation_tsa(args)
                elif args.agent == 'transfer_ppo':
                    transfer_ppo_tsa(args)
                elif args.agent == 'transfer_a2c':
                    transfer_a2c_tsa(args)
                elif args.agent == 'transfer_distral':
                    transfer_distral_tsa(args)
                elif args.agent == 'PI':
                    ppo_pixel_PI(args)
                elif args.agent == 'fc_discrete':
                    fc_discrete(args)
                elif args.agent == 'nmf_sample':
                    nmf_sample(args)
                exp_finished = True
        except Exception as e:
            traceback.print_exc()
        finally:
            if not exp_finished:
                push_args(args_str, exp_path)
                break # should quit immediately
