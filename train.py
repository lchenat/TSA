#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
from ipdb import slaunch_ipdb_on_exception
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, required=True)

args = parser.parse_args()


def option_ciritc_pixel_atari(name):
    config = Config()
    log_dir = get_default_log_dir(option_ciritc_pixel_atari.__name__)
    config.task_fn = lambda: Task(name, log_dir=log_dir, num_envs=config.num_workers)
    config.eval_env = Task(name, episode_life=False)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: OptionCriticNet(NatureConvBody(), config.action_dim, num_options=4)
    config.random_option_prob = LinearSchedule(0.1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    config.logger = get_logger(tag=option_ciritc_pixel_atari.__name__)
    run_steps(OptionCriticAgent(config))

def ppo_pixel_atari(name):
    config = Config()
    log_dir = get_default_log_dir(ppo_pixel_atari.__name__)
    config.task_fn = lambda: Task(name, log_dir=log_dir, num_envs=config.num_workers)
    config.eval_env = Task(name, episode_life=False)
    config.num_workers = 8
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
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
    config.logger = get_logger(tag=ppo_pixel_atari.__name__)
    run_steps(PPOAgent(config))

### tsa ###

def ppo_pixel_tsa():
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
    # config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, TSAMiniConvBody())

    config.state_dim = 512
    config.n_abs = 50 # number of abstract state
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

### end of tsa ###

def plot():
    import matplotlib.pyplot as plt
    plotter = Plotter()
    dirs = [
        'a2c_pixel_atari-181026-160814',
        'dqn_pixel_atari-181026-160501',
        'n_step_dqn_pixel_atari-181026-160906',
        'option_ciritc_pixel_atari-181026-160931',
        'ppo_pixel_atari-181028-092202',
        'quantile_regression_dqn_pixel_atari-181026-160630',
        'categorical_dqn_pixel_atari-181026-160743',
    ]
    names = [
        'A2C',
        'DQN',
        'NStepDQN',
        'OptionCritic',
        'PPO',
        'QRDQN',
        'C51'
    ]

    plt.figure(0)
    for i, dir in enumerate(dirs):
        data = plotter.load_results(['./data/benchmark/%s' % (dir)], episode_window=100)
        x, y = data[0]
        plt.plot(x, y, label=names[i])
    plt.xlabel('steps')
    plt.ylabel('episode return')
    plt.legend()
    plt.savefig('./images/breakout.png')

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed(0)
    select_device(-1)

    with slaunch_ipdb_on_exception():
        ppo_pixel_tsa()

    # select_device(0)
