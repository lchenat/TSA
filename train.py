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
import json
import copy
import socket
import traceback
import shutil

git_sha = get_git_sha()

def _command_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('op', type=str, choices=['new', 'join', 'push'],
        help='create a new exp with name exp-tag or join an old one, or wait for one to solve')
    parser.add_argument('exp', type=str, default='exps/exp', 
        help='path of the experiment file')
    parser.add_argument('-d', action='store_true') # debug mode
    parser.add_argument('--tag', type=str, default='0')
    parser.add_argument('--skip', action='store_true') 
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
            'imitation',
            'fc_discrete', # use fc body, tabular cases
            'fc_continuous', # use fc body
            'nmf_sample', # non-tabular cases
        ],
    )
    # environment (the task setting, first level directory)
    task = parser.add_argument_group('task')
    task.add_argument('--env', default='pick', choices=['pick', 'reach', 'grid', 'reacher'])
    ## gridworld only
    task.add_argument('-l', type=int, default=16)
    task.add_argument('-T', type=int, default=100)
    task.add_argument('--window', type=int, default=1)
    task.add_argument('--obs_type', default='rgb', choices=['rgb', 'mask'])
    ##
    task.add_argument('--env_config', type=str, default='data/env_configs/pick/map49-n_goal-2-min_dis-4')
    task.add_argument('--goal_fn', type=str, default='data/goals/fourroom/9_9')
    task.add_argument('--scale', type=int, default=1)
    task.add_argument('--task_length', type=int, default=1)
    task.add_argument('--normalized_reward', action='store_true')
    ## simple_grid only
    task.add_argument('--map_name', type=str, default='fourroom')
    ## reacher only
    task.add_argument('--n_bins', type=int, nargs=2, default=[0, 0])
    task.add_argument('--no_goal', action='store_true')
    task.add_argument('--sparse', action='store_true')
    ##
    task.add_argument('--discount', type=float, default=0.99)
    task.add_argument('--min_dis', type=int, default=1)
    task.add_argument('--task_config', type=str, default=None) # read from file
    task.add_argument('--expert', choices=['hand_coded', 'nineroom', 'fourroom_q'], default='hand_coded')
    task.add_argument('--expert_fn', type=str, default=None)
    # network
    algo.add_argument('--visual', choices=['minimini', 'mini', 'normal'], default='mini')
    algo.add_argument('--feat_dim', type=int, default=512)
    algo.add_argument('--gate', default='relu', choices=['relu', 'softplus', 'lrelu'])
    algo.add_argument('--net', default='prob', choices=['prob', 'fc', 'vq', 'pos', 'sample', 'baseline', 'map', 'imap'])
    algo.add_argument('--n_abs', type=int, default=512)
    algo.add_argument('--abs_fn', type=str, default=None)
    algo.add_argument('--res_config', type=str, default=None) # load the network config
    algo.add_argument('--actor', choices=['linear', 'nonlinear', 'split'], default='linear')
    algo.add_argument('--rate', type=float, default=1)
    algo.add_argument('--rollout_length', type=int, default=128) # works for PPO only
    algo.add_argument('--batch_size', type=int, default=32)
    algo.add_argument('--num_workers', type=int, default=8)
    algo.add_argument('--imitate_loss', choices=['kl', 'mse'], default='kl')
    algo.add_argument('--final_eps', type=float, default=0.01)
    ## simple grid only
    algo.add_argument('--hidden', type=int, nargs='+', default=(16,))
    algo.add_argument('--sample_fn', type=str, default=None) # only currently, it is actually general
    # network setting
    algo.add_argument('--label', choices=['action', 'abs'], default='action')
    algo.add_argument('--weight', type=str, default=None)
    algo.add_argument('--load_part', default='all', choices=['all', 'abs'])
    algo.add_argument('--fix_abs', action='store_true')
    algo.add_argument('--temperature', type=float, nargs='+', default=[1.0])
    # NMF sample only
    algo.add_argument('--kl_coeff', type=float, default=1.0) # for nmf_sample
    algo.add_argument('--abs_coeff', type=float, default=1.0)
    algo.add_argument('--abs_mean', type=float, default=None)
    algo.add_argument('--abs_mode', choices=['mse', 'kl'], default='mse')
    algo.add_argument('--load_actor', action='store_true')
    # optimization
    algo.add_argument('--opt', choices=['vanilla'], default='vanilla')
    algo.add_argument('-lr', type=float, default=0.00025)
    algo.add_argument('--weight_decay', type=float, default=0.0)
    algo.add_argument('--momentum', type=float, default=0.0)
    algo.add_argument('--algo_config', type=str, default=None) # read from file
    # others, should not affect performance (except seed)
    parser.add_argument('--mode', default='train', choices=['train', 'save_abs'])
    parser.add_argument('--eval_interval', type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--tag', type=str, default=None)
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

def get_env_config(args):
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config['min_dis'] = args.min_dis
        if args.normalized_reward:
            env_config['reward_config'] = {'wall_penalty': -0.01, 'time_penalty': -0.01, 'complete_sub_task': 0.1, 'complete_all': 1, 'fail': -1}
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
            scale=args.scale,
            obs_type=args.obs_type,
            task_length=args.task_length, # for pick only
        )
    return env_config

# self-defined logname function
# return task name, algo name and the rest
def get_log_tags(args):
    tags = dict()
    if args.task_config:
        tags['task'] = Path(args.task_config).stem
    else:
        if args.env in ['pick', 'reach']:
            tags['task'] = '.'.join([
                args.env,
                args.obs_type,
                Path(args.env_config).name,
                'min_dis-{}'.format(args.min_dis),
            ])
        elif args.env == 'grid':
            tags['task']='.'.join([
                args.env,
                env_config['main']['map_name'],
                #'{}_{}'.format(*env_config['main']['goal_locs']),
                Path(args.goal_fn).name,
                'md-{}'.format(args.min_dis),
                'T-{}'.format(args.T),
            ]),
        elif args.env == 'reacher':
            env_name = args.env
            if args.no_goal: env_name += '.ng'
            if args.sparse: env_name += '.sp'
            tags['task'] = '.'.join([
                env_name,
                Path(args.goal_fn).name,
            ])
    if args.algo_config:
        tags['algo'] = Path(args.algo_config).stem
    else:
        tags['algo'] = args.algo_name
    tags['others'] = parse_tag(args.tag, args)
    tags['seed'] = args.seed
    return tags

def set_optimizer_fn(args, config):
    def optimizer_fn(model):
        params = filter(lambda p: p.requires_grad, model.parameters())
        return VanillaOptimizer(
            params, 
            torch.optim.RMSprop(
                params,
                lr=args.lr,
                momentum=args.momentum,
                alpha=0.99,
                eps=1e-5,
                weight_decay=args.weight_decay), 
            config.gradient_clip
        )
    config.optimizer_fn = optimizer_fn

def process_temperature(temperature):
    if len(temperature) == 1:
        temperature = linspace(temperature[0], temperature[0], 2, repeat_end=True)
    elif len(temperature) == 3:
        temperature[2] = int(temperature[2])
        temperature = linspace(*temperature, repeat_end=True)
    else:
        raise Exception('this length is not gonna work')
    return temperature

def get_gate(gate):
    if gate == 'relu':
        return F.relu
    elif gate == 'softplus':
        return F.softplus
    elif gate == 'lrelu':
        return F.leaky_relu

def get_visual_body(args, config):
    if args.obs_type == 'rgb':
        n_channels = 3
    else:
        n_channels = config.eval_env.observation_space.shape[0]
    if args.visual == 'mini':
        visual_body = TSAMiniConvBody(n_channels*config.env_config['main']['window'], feature_dim=args.feat_dim, scale=args.scale, gate=get_gate(args.gate))
    elif args.visual == 'normal':
        visual_body = TSAConvBody(n_channels*config.env_config['main']['window'], feature_dim=args.feat_dim, scale=args.scale, gate=get_gate(args.gate)) 
    return visual_body

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
            elif hasattr(network, 'network'):
                load_filter = lambda x: x.startswith('network.phi_body')
            elif hasattr(network, 'body'):
                load_filter = lambda x: x.startswith('body')
            else:
                raise Exception('unsupported network type')
        loaded_weight_dict = {k: v for k, v in torch.load(
            args.weight,
            map_location=lambda storage, loc: storage)['network'].items()
            if ((k in weight_dict) and load_filter(k))}
        print('loaded weight keys: {}'.format(loaded_weight_dict.keys()))
        weight_dict.update(loaded_weight_dict)
        network.load_state_dict(weight_dict)
        if 'action_predictor' in weight_dict:
            config.action_predictor.load_state_dict(weight_dict['action_predictor'])
    if args.fix_abs:
        if hasattr(network, 'abs_encoder'):
            for p in network.abs_encoder.parameters():
                p.requires_grad = False
        elif hasattr(network, 'network'): # categoricalactorcriticnet
            for p in network.network.phi_body.parameters():
                p.requires_grad = False
        elif hasattr(network, 'body'): # qnet
            for p in network.body.parameters():
                p.requires_grad = False
        else:
            raise Exception('unsupported network type')

def process_goals(goal_fn): # should be read json what are you doing?
    with open(goal_fn) as f:
        goals = json.load(f)
    return [tuple(goal) for goal in goals]

# for normal tsa
def get_network(visual_body, args, config):
    if args.net == 'baseline':
        if args.actor == 'linear':
            actor = MultiLinear(visual_body.feature_dim, config.action_dim, config.eval_env.n_tasks, key='task_id', w_scale=1e-3)
        elif args.actor == 'nonlinear': # gate
            actor = MultiMLP(visual_body.feature_dim, tuple(args.hidden) + (config.action_dim,), config.eval_env.n_tasks, key='task_id', w_scale=1e-3)
        else:
            raise Exception('unsupported actor')
        algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
        network = CategoricalActorCriticNet(
            config.eval_env.n_tasks,
            config.state_dim,
            config.action_dim, 
            visual_body,
            actor=actor,
        )
    else:
        if args.net == 'vq':
            algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
            abs_encoder = VQAbstractEncoder(args.n_abs, config.abs_dim, visual_body)
            if args.actor == 'nonlinear':
                actor = NonLinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
            else:
                actor = LinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'fc':
            algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
            abs_encoder = FCAbstractEncoder(args.n_abs, visual_body)
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'prob':
            algo_name = '.'.join([args.agent, args.net, 'n_abs-{}'.format(args.n_abs)])
            temperature = process_temperature(args.temperature)
            abs_encoder = ProbAbstractEncoder(args.n_abs, visual_body, temperature=temperature)
            if args.actor == 'linear':
                actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
            elif args.actor == 'nonlinear':
                actor = NonLinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks, tuple(args.hidden)) # args.dim is only used here!
            else:
                raise Exception('unknown actor net')
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
        else:
            raise Exception('unsupported network type')
        critic_body = visual_body
        critic = TSACriticNet(critic_body, config.eval_env.n_tasks)
        network = TSANet(config.action_dim, abs_encoder, actor, critic)
    return network, algo_name

def get_grid_network(args, config):
    n_tasks = config.eval_env.n_tasks
    algo_name = [args.agent, args.net]
    if args.net == 'baseline':
        network = OldCategoricalActorCriticNet(
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
        fc_body = FCBody(config.state_dim, hidden_units=tuple(args.hidden))
        temperature = process_temperature(args.temperature)
        abs_encoder = ProbAbstractEncoder(args.n_abs, fc_body, temperature=temperature)
        if args.actor == 'linear':
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.actor == 'nonlinear':
            actor = NonLinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
    else:
        raise Exception('unsupported network')
    critic_body = abs_encoder # does this matter?
    critic = TSACriticNet(critic_body, config.eval_env.n_tasks)
    network = TSANet(config.action_dim, abs_encoder, actor, critic)
    return network, '.'.join(algo_name)

def get_reacher_network(args, config):
    n_tasks = config.eval_env.n_tasks
    algo_name = [args.agent, args.net]
    phi_body = FCBody(
        config.state_dim, 
        hidden_units=tuple(args.hidden)
    )
    if args.net == 'baseline':
        if args.actor == 'split':
            actor = SplitBody(
                MultiLinear(phi_body.feature_dim, config.action_dim.sum(), n_tasks, key='task_id', w_scale=1e-3),
                2, # here assume half split
            )
        else:
            actor = None
        network = CategoricalActorCriticNet(
            n_tasks,
            config.state_dim,
            config.action_dim.prod(),
            phi_body,
            actor=actor,
        )
        return network, '.'.join(algo_name)
    elif args.net == 'gaussian':
        network = GaussianActorCriticNet(
            n_tasks,
            config.state_dim,
            config.action_dim,
            phi_body=phi_body,
        )
    else:
        raise Exception('unsupported network')
    return network, '.'.join(algo_name)

def get_expert(args, config):
    if args.expert == 'hand_coded':
        return None
    elif args.expert == 'nineroom':
        with open(args.expert_fn) as f:
            expert_fns = json.load(f)
        experts = dict()
        for index, weight_path in expert_fns.items():
            visual_body = TSAMiniConvBody(
                config.eval_env.observation_space.shape[0], 
                512,
                scale=2,
                #gate=F.softplus,
            )
            expert = CategoricalActorCriticNet(
                config.eval_env.n_tasks,
                config.state_dim,
                config.action_dim,
                visual_body,
            )
            # load weight
            weight_dict = expert.state_dict()
            loaded_weight_dict = {k: v for k, v in torch.load(
                weight_path,
                map_location=lambda storage, loc: storage)['network'].items()
                if k in weight_dict}
            weight_dict.update(loaded_weight_dict)
            expert.load_state_dict(weight_dict)
            experts[int(index)] = expert
        return experts
    elif args.expert == 'fourroom_q':
        with open(args.expert_fn) as f:
            expert_fns = json.load(f)
        experts = dict()
        for index, weight_path in expert_fns.items():
            visual_body = TSAMiniConvBody(
                config.eval_env.observation_space.shape[0], 
                512,
                scale=2,
            )
            expert = QNet(config.eval_env.n_tasks, config.action_dim, visual_body)
            # load weight
            weight_dict = expert.state_dict()
            loaded_weight_dict = {k: v for k, v in torch.load(
                weight_path,
                map_location=lambda storage, loc: storage)['network'].items()
                if k in weight_dict}
            weight_dict.update(loaded_weight_dict)
            expert.load_state_dict(weight_dict)
            experts[int(index)] = expert
        return experts
    else:
        raise Exception('unsupported expert type')

def ppo_pixel_tsa(args):
    config = Config()
    env_config = get_env_config(args)
    config.env_config = env_config
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = args.num_workers
    visual_body = get_visual_body(args, config)
    network, args.algo_name = get_network(visual_body, args, config)
    config.network_fn = lambda: network
    process_weight(network, args, config)
    set_optimizer_fn(args, config)
    if args.obs_type == 'rgb':
        assert args.env in ['pick', 'reach']
        config.state_normalizer = ImageNormalizer() # tricky
    config.discount = args.discount
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = args.rollout_length
    config.optimization_epochs = 3
    config.mini_batch_size = args.batch_size * config.num_workers
    config.ppo_ratio_clip = 0.1
    config.log_interval = args.rollout_length * config.num_workers
    config.max_steps = 1e4 if args.d else int(2e6)
    if args.steps is not None: config.max_steps = args.steps
    config.eval_interval = args.eval_interval
    config.save_interval = args.save_interval
    if args.mode == 'train':
        config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
        config.logger.add_text('Configs', [{
            'git sha': git_sha,
            **vars(args),
            }])
        return PPOAgent(config), run_steps
    else:
        config.abs_save_path = Path(args.weight)
        agent = PPOAgent(config)
        return agent, save_abs

def fc_discrete(args):
    config = Config()
    config.num_workers = 5
    if args.env == 'grid':
        goal_locs = process_goals(args.goal_fn)
        env_config = dict(
            main=dict(
                map_name=args.map_name,
                goal_locs=goal_locs,
                min_dis=args.min_dis,
            ),
            T=args.T, # 250?
        )
        config.task_fn = lambda: DiscreteGridTask(env_config, num_envs=config.num_workers)
        config.eval_env = DiscreteGridTask(env_config)
        network, args.algo_name = get_grid_network(args, config)
    elif args.env == 'reacher':
        with open(args.goal_fn) as f:
            goal_dict = json.load(f)
        env_config = dict(
            main=dict(
                goals=goal_dict['goals'],
                sample_indices=goal_dict['sample_indices'],
                n_bins=args.n_bins,
                with_goal_pos=not args.no_goal,
                sparse=args.sparse,
            ),
            T=args.T,
        )
        config.task_fn = lambda: ReacherTask(env_config, num_envs=config.num_workers)
        config.eval_env = ReacherTask(env_config)
        network, args.algo_name = get_reacher_network(args, config)
    else:
        raise Exception('unsupported environment')

    config.network_fn = lambda: network
    def optimizer_fn(model):
        params = filter(lambda p: p.requires_grad, model.parameters())
        return VanillaOptimizer(params, torch.optim.RMSprop(params, 0.001), config.gradient_clip)
    config.optimizer_fn = optimizer_fn
    #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    process_weight(network, args, config) # interesting
    config.discount = args.discount
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.rollout_length = args.rollout_length
    config.optimization_epochs = 10
    config.mini_batch_size = 32 * 5
    config.ppo_ratio_clip = 0.2
    config.log_interval = args.rollout_length * config.num_workers * 10 # may skip 10

    config.max_steps = 10000 if args.d else 180000
    if args.steps is not None: config.max_steps = args.steps
    config.eval_interval = 5 # 50
    config.save_interval = args.save_interval
    config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': git_sha,
        **vars(args),
        }])
    return PPOAgent(config), run_steps

def nmf_sample(args):
    config = Config()
    assert args.sample_fn is not None, 'need args.sample_fn'
    with open(args.sample_fn, 'rb') as f:
        sample_dict = dill.load(f) # abs, policy
        args.n_abs = sample_dict['abs'].shape[1] # here change the n_abs!!!
        n_tasks = len(sample_dict['policies'])
    config.sample_dict = sample_dict
    config.load_actor = args.load_actor
    if args.env == 'grid':
        goal_locs = process_goals(args.goal_fn)
        env_config = dict(
            main=dict(
                map_name=args.map_name,
                goal_locs=goal_locs,
                min_dis=args.min_dis,
            ),
            T=args.T, # 250?
        )
        config.eval_env = DiscreteGridTask(env_config)
        network, args.algo_name = get_grid_network(args, config)
    elif args.env == 'reacher':
        with open(args.goal_fn) as f:
            goal_dict = json.load(f)
        env_config = dict(
            main=dict(
                goals=goal_dict['goals'],
                sample_indices=goal_dict['sample_indices'],
                n_bins=args.n_bins,
                with_goal_pos=not args.no_goal,
                sparse=args.sparse,
            ),
            T=args.T,
        )
        config.eval_env = ReacherTask(env_config)
        network, args.algo_name = get_reacher_network(args, config)
    elif args.env == 'pick':
        env_config = get_env_config(args)
        config.env_config = env_config
        config.eval_env = Task(env_config)
        visual_body = get_visual_body(args, config)
        network, args.algo_name = get_network(visual_body, args, config)
        if args.obs_type == 'rgb':
            config.state_normalizer = ImageNormalizer() # tricky
    else:
        raise Exception('unsupported env')
    process_weight(network, args, config)
    config.network_fn = lambda: network
    def optimizer_fn(model):
        params = filter(lambda p: p.requires_grad, model.parameters())
        return VanillaOptimizer(params, torch.optim.RMSprop(params, 0.001), config.gradient_clip)
    config.kl_coeff = args.kl_coeff
    config.abs_coeff = args.abs_coeff
    config.abs_mean = args.abs_mean
    config.abs_mode = args.abs_mode
    config.optimizer_fn = optimizer_fn
    #config.gradient_clip = 5
    config.batch_size = 32 * n_tasks
    config.log_interval = config.batch_size * 10

    config.max_steps = 300000 if args.d else 1200000
    if args.steps is not None: config.max_steps = args.steps
    config.eval_interval = 5
    config.save_interval = args.save_interval
    log_tags = dict(
        task='.'.join([args.env, Path(args.sample_fn).name]),
        algo=args.algo_name,
        others=parse_tag(args.tag, args),
        seed=args.seed,
    )
    config.logger = get_logger(args.hash_code, tags=log_tags, skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': git_sha,
        **vars(args),
        }])
    return NMFAgent(config), run_supervised_steps

def imitation_tsa(args):
    config = Config()
    config.imitate_loss = args.imitate_loss
    env_config = get_env_config(args)
    config.env_config = env_config
    config.task_fn = lambda: Task(env_config, num_envs=config.num_workers)
    config.eval_env = Task(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.expert = get_expert(args, config)
    config.num_workers = 8
    visual_body = get_visual_body(args, config)
    network, args.algo_name = get_network(visual_body, args, config)
    config.network_fn = lambda: network
    process_weight(network, args, config)
    set_optimizer_fn(args, config)
    if args.obs_type == 'rgb':
        assert args.env in ['pick', 'reach']
        config.state_normalizer = ImageNormalizer() # tricky
    config.discount = args.discount
    #config.gradient_clip = 0.5
    config.rollout_length = args.rollout_length
    config.log_interval = config.num_workers * config.rollout_length
    config.max_steps = 300000 if args.d else 1200000
    if args.steps is not None: config.max_steps = args.steps
    config.eval_interval = args.eval_interval
    config.save_interval = 1 # in terms of eval interval
    config.logger = get_logger(args.hash_code, tags=get_log_tags(args), skip=args.skip)
    config.logger.add_text('Configs', [{
        'git sha': git_sha,
        **vars(args),
        }])
    return ImitationAgent(config), run_steps

def run_exp(exp_path, parser, command_args):
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
            args.d = command_args.d # pass debug flag
            args.skip = command_args.skip
            global Task # fix
            if args.env == 'pick':
                Task = PickGridWorldTask
            elif args.env == 'reach':
                Task = ReachGridWorldTask
            elif args.env == 'grid':
                Task = DiscreteGridTask
            elif args.env == 'reacher':
                Task = ReacherTask

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
                    agent, run_f = ppo_pixel_tsa(args)
                elif args.agent == 'baseline':
                    agent, run_f = ppo_pixel_baseline(args)
                elif args.agent == 'imitation':
                    agent, run_f = imitation_tsa(args)
                elif args.agent == 'PI':
                    agent, run_f = ppo_pixel_PI(args)
                elif args.agent == 'fc_discrete':
                    agent, run_f = fc_discrete(args)
                elif args.agent == 'nmf_sample':
                    agent, run_f = nmf_sample(args)
                elif args.agent == 'nmf_reg':
                    agent, run_f = nmf_reg(args)
                elif args.agent == 'dqn':
                    agent, run_f = dqn(args)
                elif args.agent == 'meta_linear_q':
                    agent, run_f = meta_linear_q(args)
                elif args.agent == 'sarsa':
                    agent, run_f = sarsa(args)
                run_f(agent)
                exp_finished = True
        except Exception as e:
            traceback.print_exc()
        finally:
            if not exp_finished:
                if not command_args.skip and 'agent' in vars() and stdin_choices('experiment is not finished, want to clean up the log?', ['y', 'n']) == 'y':
                    agent.config.logger.clear() # when agent is not defined, this cannot be ran
                push_args(args_str, exp_path)
                break

def main(args=None):
    command_args = _command_parser().parse_args(args)
    parser = _exp_parser()
    if command_args.op == 'new':
        assert Path(command_args.exp).suffix != '.run', '.run file should be joined instead of new'
        exp_path = Path('{}-{}.run'.format(command_args.exp, command_args.tag))
        if not exp_path.exists() or stdin_choices('{} exists, want to replace?'.format(exp_path), ['y', 'n']):
            shutil.copy(command_args.exp, str(exp_path))
        else: exit()
    elif command_args.op == 'join':
        exp_path = Path(command_args.exp)
        assert exp_path.suffix == '.run', 'only support run filetype, name: {}, suffix: {}'.format(exp_path, exp_path.suffix)
    elif command_args.op == 'push': # push exp_path and hashcode (not the same hashcode should not be run)
        raise Exception('not finished yet')
    else:
        raise Exception('unsupported operation')
    if not command_args.d and is_git_diff():
        print(colored('please commit your changes before running new experiments!', 'red', attrs=['bold']))
        exit() # end the program
    run_exp(exp_path, parser, command_args)

if __name__ == '__main__':
    main()
