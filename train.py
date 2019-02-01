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
import copy

def _command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', default='tsa', choices=['tsa', 'baseline', 'supervised', 'imitation', 'transfer_a2c', 'transfer_ppo', 'transfer_distral'])
    # environment
    parser.add_argument('--env', default='pick', choices=['pick', 'reach'])
    parser.add_argument('-l', type=int, default=16)
    parser.add_argument('-T', type=int, default=100)
    parser.add_argument('--window', type=int, default=1)
    #parser.add_argument('--env_config', type=str, default='data/env_configs/map49-single')
    parser.add_argument('--env_config', type=str, default='data/env_configs/pick/map49-n_goal-2-min_dis-4')
    parser.add_argument('--discount', type=float, default=0.99)
    # network
    parser.add_argument('--visual', choices=['mini', 'normal', 'large'], default='mini')
    parser.add_argument('--net', default='prob', choices=['prob', 'vq', 'pos', 'kv', 'id', 'sample', 'baseline', 'i2a', 'bernoulli'])
    parser.add_argument('--n_abs', type=int, default=512)
    parser.add_argument('--abs_fn', type=str, default=None)
    parser.add_argument('--actor', choices=['linear', 'nonlinear'], default='nonlinear')
    parser.add_argument('--critic', default='visual', choices=['critic', 'abs'])
    # transfer network
    parser.add_argument('--t_net', default='prob', choices=['prob', 'vq', 'pos', 'kv', 'id', 'sample', 'baseline', 'i2a', 'bernoulli'])
    parser.add_argument('--t_n_abs', type=int, default=512)
    parser.add_argument('--t_abs_fn', type=str, default=None)
    parser.add_argument('--t_actor', choices=['linear', 'nonlinear'], default='nonlinear')
    parser.add_argument('--distill_w', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.0)
    # network setting
    parser.add_argument('--label', choices=['action', 'abs'], default='action')
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--fix_abs', action='store_true')
    parser.add_argument('--temperature', type=float, nargs='+', default=[1.0])
    # aux loss
    parser.add_argument('--pred_action', action='store_true')
    parser.add_argument('--recon', action='store_true')
    parser.add_argument('--trans', action='store_true')
    parser.add_argument('--reg_abs_fn', type=str, default=None)
    # optimization
    parser.add_argument('--opt', choices=['vanilla', 'alt', 'diff'], default='vanilla')
    parser.add_argument('--opt_gap', nargs=2, type=int, default=[9, 9])
    parser.add_argument('-lr', nargs='+', type=float, default=[0.00025])
    # others
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('-d', action='store_true')
    return parser

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
        config.reg_abs = RegAbs(visual_body, abs_dict)
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
        loaded_weight_dict = {k: v for k, v in torch.load(args.weight, map_location=lambda storage, loc: storage).items() if k in weight_dict}
        weight_dict.update(loaded_weight_dict)
        network.load_state_dict(weight_dict)
        if 'action_predictor' in weight_dict:
            config.action_predictor.load_state_dict(weight_dict['action_predictor'])
    if args.fix_abs:
        for p in network.abs_encoder.parameters():
            p.requires_grad = False

def get_network(visual_body, args, config):
    if args.net == 'baseline':
        log_name = '{}-{}-{}'.format(args.agent, args.net, lastname(args.env_config))
        network = CategoricalActorCriticNet(
            config.eval_env.n_tasks,
            config.state_dim,
            config.action_dim, 
            visual_body,
            actor_body=FCBody(visual_body.feature_dim, (args.n_abs,)),
        )
    else:
        if args.net == 'vq':
            log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), args.n_abs)
            abs_encoder = VQAbstractEncoder(args.n_abs, config.abs_dim, visual_body)
            if args.actor == 'nonlinear':
                actor = NonLinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
            else:
                actor = LinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'prob':
            log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), args.n_abs)
            temperature = process_temperature(args.temperature)
            abs_encoder = ProbAbstractEncoder(args.n_abs, visual_body, temperature=temperature)
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'pos':
            assert args.abs_fn is not None, 'need args.abs_fn'
            with open(args.abs_fn, 'rb') as f:
                abs_dict = dill.load(f)
                n_abs = len(set(abs_dict[0].values())) # only have 1 map!
            log_name = '{}-{}-{}-{}'.format(args.agent, args.net, lastname(args.env_config), lastname(args.abs_fn)[:-4])
            print(abs_dict)
            abs_encoder = PosAbstractEncoder(n_abs, abs_dict)
            actor = LinearActorNet(n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'kv': # key-value
            log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), args.n_abs)
            abs_encoder = KVAbstractEncoder(args.n_abs, config.abs_dim, visual_body)
            if args.actor == 'nonlinear':
                actor = NonLinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
            else:
                actor = LinearActorNet(config.abs_dim, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'id':
            temperature = process_temperature(args.temperature)
            n_abs = config.action_dim
            log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), n_abs)
            abs_encoder = ProbAbstractEncoder(n_abs, visual_body, temperature=temperature)
            actor = IdentityActor()
        elif args.net == 'sample':
            temperature = process_temperature(args.temperature)
            log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), args.n_abs)
            abs_encoder = SampleAbstractEncoder(args.n_abs, visual_body, temperature=temperature)
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'i2a':
            temperature = process_temperature(args.temperature)
            log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), args.n_abs)
            abs_encoder = I2AAbstractEncoder(args.n_abs, visual_body, temperature=temperature)
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        elif args.net == 'bernoulli':
            temperature = process_temperature(args.temperature)
            log_name = '{}-{}-{}-n_abs-{}'.format(args.agent, args.net, lastname(args.env_config), args.n_abs)
            abs_encoder = BernoulliAbstractEncoder(args.n_abs, visual_body, temperature=temperature)
            actor = LinearActorNet(args.n_abs, config.action_dim, config.eval_env.n_tasks)
        if args.critic == 'visual':
            critic_body = visual_body
        elif args.critic == 'abs':
            critic_body = abs_encoder
        critic = TSACriticNet(critic_body, config.eval_env.n_tasks)
        network = TSANet(config.action_dim, abs_encoder, actor, critic)
    return network, log_name

def ppo_pixel_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 8
    visual_body = get_visual_body(args, config)
    network, config.log_name = get_network(visual_body, args, config)
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
    config.logger.save_file(env_config['main'], 'env_config')
    run_steps(PPOAgent(config))

def ppo_pixel_baseline(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        n_tasks = len(env_config['train_combos'] + env_config['test_combos'])
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
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
    config.logger.save_file(env_config['main'], 'env_config')
    run_steps(PPOAgent(config))

def supervised_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        n_tasks = len(env_config['train_combos'] + env_config['test_combos'])
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
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
    network, config.log_name = get_network(visual_body, args, config)
    config.network_fn = lambda: network
    set_aux_network(visual_body, args, config)
    process_weight(network, args, config)
    #config.network_fn = lambda: CategoricalActorCriticNet(n_tasks, config.state_dim, config.action_dim, TSAMiniConvBody(3*env_config['window'])) # debug
    config.state_normalizer = ImageNormalizer()
    config.discount = args.discount
    config.log_interval = 1
    config.max_steps = 10000 if args.d else int(10000)
    config.save_interval = 1 # how many steps to save a model
    config.eval_interval = 100
    config.log_name += '-label-{}'.format(args.label)
    if args.tag: config.log_name += '-{}'.format(args.tag)
    config.logger = get_logger(tag=config.log_name)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    config.logger.save_file(env_config['main'], 'env_config')
    run_supervised_steps(SupervisedAgent(config))

def imitation_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 8
    visual_body = get_visual_body(args, config)
    network, config.log_name = get_network(visual_body, args, config)
    config.network_fn = lambda: network
    set_aux_network(visual_body, args, config)
    process_weight(network, args, config)
    set_optimizer_fn(args, config)
    config.state_normalizer = ImageNormalizer()
    config.discount = args.discount
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.log_interval = config.num_workers * config.rollout_length
    config.eval_interval = config.log_interval * 100
    config.max_steps = 1e4 if args.d else int(1.5e7)
    config.save_interval = 0 # how many steps to save a model
    if args.tag: config.log_name += '-{}'.format(args.tag)
    config.logger = get_logger(tag=config.log_name)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    config.logger.save_file(env_config['main'], 'env_config')
    run_steps(ImitationAgent(config))

# TODO:
# you need to set log_name
def transfer_ppo_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 8
    # source: the one with abstraction, try to process weight
    visual_body = get_visual_body(args, config)
    network, config.log_name = get_network(visual_body, args, config)
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
    config.rollout_length = 128
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.max_steps = 1.5e7 if args.d else int(1.5e7)
    config.save_interval = 0 # how many steps to save a model
    config.log_name += '-w-{}'.format(config.distill_w)
    if args.tag: config.log_name += '-{}'.format(args.tag)
    config.logger = get_logger(tag=config.log_name)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    config.logger.save_file(env_config['main'], 'env_config')
    run_steps(TransferPPOAgent(config))

def transfer_a2c_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 16
    visual_body = get_visual_body(args, config)
    network, config.log_name = get_network(visual_body, args, config)
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
    config.save_interval = 0 # how many steps to save a model
    config.log_name += '-w-{}'.format(config.distill_w)
    if args.tag: config.log_name += '-{}'.format(args.tag)
    config.logger = get_logger(tag=config.log_name)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    config.logger.save_file(env_config['main'], 'env_config')
    run_steps(TransferA2CAgent(config))

def transfer_distral_tsa(args):
    config = Config()
    with open(args.env_config, 'rb') as f:
        env_config = dill.load(f)
        env_config['window'] = args.window
        env_config = dict(
            main=env_config,
            l=args.l,
            T=args.T,
        )
        config.env_config = env_config
    config.task_fn = lambda: GridWorldTask(env_config, num_envs=config.num_workers)
    config.eval_env = GridWorldTask(env_config)
    print('n_tasks:', config.eval_env.n_tasks)
    config.num_workers = 16
    visual_body = get_visual_body(args, config)
    network, config.log_name = get_network(visual_body, args, config)
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
    config.save_interval = 0 # how many steps to save a model
    config.log_name += '-w-{}'.format(config.distill_w)
    if args.tag: config.log_name += '-{}'.format(args.tag)
    config.logger = get_logger(tag=config.log_name)
    config.logger.add_text('Configs', [{
        'git sha': get_git_sha(),
        **vars(args),
        }])
    config.logger.save_file(env_config['main'], 'env_config')
    run_steps(TransferDistralAgent(config))


if __name__ == '__main__':
    parser = _command_line_parser()
    args = parser.parse_args()
    if not args.d and is_git_diff():
        print(colored('please commit your changes before running new experiments!', 'red', attrs=['bold']))
        exit()
    if args.env == 'pick':
        GridWorldTask = PickGridWorldTask
    elif args.env == 'reach':
        GridWorldTask = ReachGridWorldTask

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
        elif args.agent == 'imitation':
            imitation_tsa(args)
        elif args.agent == 'transfer_ppo':
            transfer_ppo_tsa(args)
        elif args.agent == 'transfer_a2c':
            transfer_a2c_tsa(args)
        elif args.agent == 'transfer_distral':
            transfer_distral_tsa(args)
