
import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from collections import deque

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from ..utils import *

try:
    import roboschool
except ImportError:
    pass

# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, seed, rank, log_dir, episode_life=True):
    def _thunk():
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)

        if log_dir is not None:
            # env = Monitor(env=env, filename=os.path.join(log_dir, str(rank)), allow_early_resets=True)
            env = bench.Monitor(env=env, filename=os.path.join(log_dir, str(rank)), allow_early_resets=True)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env

    return _thunk


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

# Allow tensorboard to record original episode return
class Monitor(bench.Monitor):
    def __init__(self, **kwargs):
        super(Monitor, self).__init__(**kwargs)
        log_dir = kwargs['filename'].replace('./log', './tf_log')
        log_dir = '/'.join(log_dir.split('/')[:-1])
        self.tf_logger = Logger(None, log_dir)
        self.tf_step = 0

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            self.tf_step += eplen
            self.tf_logger.add_scalar('return', eprew)
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(epinfo)
                self.f.flush()
            info['episode'] = epinfo
        self.total_steps += 1
        return (ob, rew, done, info)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# I modify this to support keywords in reset
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None
        self.reset_kwargs = dict()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset(**self.reset_kwargs)
                info = self.envs[i].unwrapped.get_info()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        info = stack_dict(info)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self, **kwargs):
        self.reset_kwargs = kwargs
        return [env.reset(**kwargs) for env in self.envs]

    def close(self):
        return

    def get_info(self):
        return stack_dict([env.unwrapped.get_info() for env in self.envs])

class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, log_dir, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)

### tsa ###
from ..gridworld import ReachGridWorld, PORGBEnv, PickGridWorld
from ..simple_grid.env import DiscreteGridWorld, SampleParameterEnv
from ..simple_grid.exemplar_env import DiscreteGridWorld, RandomInitEnv, RandomGoalEnv
from ..reacher.env import MultiGoalReacherEnv, DiscretizeActionEnv

class LastWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def __getattribute__(self, attr):
        if attr == 'env':
            return object.__getattribute__(self, attr)
        env = self.env
        while True:
            if hasattr(env, attr):
                return getattr(env, attr)
            if env.unwrapped == env: break
            env = env.env 

class FiniteHorizonEnv(gym.Wrapper):
    def __init__(self, env, T=100000000):
        super().__init__(env)
        self.T = T

    def reset(self, *args, **kwargs):
        self.t = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        o, r, done, info = self.env.step(action)
        self.t += 1
        if self.t >= self.T:
            done = True
        return o, r, done, info

    @property
    def last(self):
        return LastWrapper(self)

def make_reach_gridworld_env(env_config, seed, rank):
    def _thunk():
        random_seed(seed)
        env = ReachGridWorld(**env_config['main'], seed=seed+rank)
        env = PORGBEnv(env, l=env_config['l'])
        env = FiniteHorizonEnv(env, T=env_config['T'])

        return env

    return _thunk

def make_pick_gridworld_env(env_config, seed, rank):
    def _thunk():
        random_seed(seed)
        env = PickGridWorld(**env_config['main'], task_length=1, seed=seed+rank)
        env = PORGBEnv(env, l=env_config['l'])
        env = FiniteHorizonEnv(env, T=env_config['T'])

        return env

    return _thunk

class ReachGridWorldTask:
    def __init__(self,
                 env_config,
                 num_envs=1,
                 seed=np.random.randint(int(1e5))):
        envs = [make_reach_gridworld_env(env_config, seed, i) for i in range(num_envs)]
        self.env = DummyVecEnv(envs)
        self.name = 'ReachGridWorld'
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape)) # state_dim is useless, it is for DummyBody which is an identity map
        self.n_maps = len(env_config['main']['map_names'])
        self.n_tasks = len(self.env.envs[0].unwrapped.g2i)
        self.env_type = 'full'

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self, index=None, train=True):
        return self.env.reset(index=index, train=train)

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)

    def get_info(self): # retrieve map_id and goal position
        return self.env.get_info()

    def get_opt_action(self):
        return [env.last.get_random_opt_action(0.99) for env in self.env.envs]

class PickGridWorldTask:
    def __init__(self,
                 env_config,
                 num_envs=1,
                 seed=np.random.randint(int(1e5))):
        envs = [make_pick_gridworld_env(env_config, seed, i) for i in range(num_envs)]
        self.env = DummyVecEnv(envs)
        self.name = 'PickGridWorld'
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape)) # state_dim is useless, it is for DummyBody which is an identity map
        self.n_maps = len(env_config['main']['map_names'])
        self.n_tasks = self.env.envs[0].unwrapped.num_obj_types
        self.env_type = 'full'

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self, index=None, train=True):
        return self.env.reset(index=index, train=train, sample_obj_pos=False)

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)

    def get_info(self): # retrieve map_id and goal position
        return self.env.get_info()

    def get_opt_action(self):
        return [env.last.get_random_opt_action(0.99) for env in self.env.envs]

def make_discrete_grid_env(env_config, seed, rank):
    def _thunk():
        random_seed(seed)
        env = RandomGoalEnv(
            RandomInitEnv(
                DiscreteGridWorld(env_config['main']['map_name'], (1, 1), (9, 9), seed=seed+rank),
                min_dis=env_config['main']['min_dis'],
            ),
            goal_locs=env_config['main']['goal_locs'],
        )
        env = FiniteHorizonEnv(env, T=env_config['T'])

        return env

    return _thunk

class DiscreteGridTask:
    def __init__(
        self,
        env_config,
        num_envs=1,
        seed=np.random.randint(int(1e5))):

        self.num_envs = num_envs
        envs = [make_discrete_grid_env(env_config, seed, i) for i in range(num_envs)]
        self.env = DummyVecEnv(envs)
        self.name = 'DiscreteGridWorld'
        self.observation_space = self.env.observation_space
        self.state_dim = len(self.env.observation_space.spaces)
        self.action_space = self.env.action_space
        self.action_dim = self.action_space.n
        self.env_type = 'simulation'
        self.n_tasks = len(self.env.envs[0].last.goals)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        next_o, r, done, info = self.env.step(actions)
        info.pop('reward_config', None)
        info['task_id'] = [env.last.goal_idx for env in self.env.envs]
        return next_o, r, done, info

    def get_info(self): # retrieve map_id and goal position
        info = self.env.get_info()
        info.pop('reward_config', None)
        info['task_id'] = [env.last.goal_idx for env in self.env.envs]
        return info

def make_reacher_env(env_config, seed, rank):
    def _thunk():
        random_seed(seed)
        env = MultiGoalReacherEnv(
                goals=env_config['main']['goals'],
                sample_indices=env_config['main']['sample_indices'],
                with_goal_pos=env_config['main']['with_goal_pos'],
                sparse=env_config['main']['sparse'],
        )
        if env_config['main']['n_bins'][0] != 0:
            env = DiscretizeActionEnv(
                env,
                n_bins=env_config['main']['n_bins'],
            )
        env = FiniteHorizonEnv(env, T=env_config['T'])

        return env

    return _thunk

class ReacherTask:
    def __init__(
        self,
        env_config,
        num_envs=1,
        seed=np.random.randint(int(1e5))):

        self.num_envs = num_envs
        envs = [make_reacher_env(env_config, seed, i) for i in range(num_envs)]
        self.env = DummyVecEnv(envs)
        self.name = 'Reacher'
        self.observation_space = self.env.observation_space
        self.state_dim = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space
        if isinstance(self.env, DiscretizeActionEnv):
            self.action_dim = self.env.action_space.nvec
        else:
            self.action_dim = self.env.action_space.shape[0]
        self.env_type = 'simulation'
        self.n_tasks = len(self.env.envs[0].last.goals)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        next_o, r, done, info = self.env.step(actions)
        info['task_id'] = [env.last.goal_idx for env in self.env.envs]
        return next_o, r, done, info

    def get_info(self): # retrieve map_id and goal position
        info = self.env.get_info()
        info['task_id'] = [env.last.goal_idx for env in self.env.envs]
        return info
