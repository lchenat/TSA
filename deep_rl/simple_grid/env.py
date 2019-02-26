# based on simple_rl's gridworld and puddle
# let's make each env a MDP, and wrapper multitask on top of that
# also note that gym is a simulation-based environment, therefore reward and transition are only based on current state, action and next state
import matplotlib.pyplot as plt
import numpy as np
from gym import Env, Wrapper, spaces
from gym.utils import seeding
from collections import namedtuple
from pathlib import Path
if __package__ == '':
    from utils import Render, GridDrawer
else:
    from .utils import Render, GridDrawer

ds_dict = {
    0: (-1, 0), # up
    1: (1, 0), # down
    2: (0, -1), # left
    3: (0, 1), # right
}

# wall: #, lava: *
def read_map(filename):
    m = []
    with open(filename) as f:
        for row in f:
            m.append(list(row.rstrip()))
    return m

# discrete gridworld
class DiscreteGridWorld(Env):
    color_list = [
        plt.cm.Blues(0.5), # agent
        plt.cm.Greys(0.0), # empty
        plt.cm.Oranges(0.95), # wall
        plt.cm.Reds(0.7), # lava
    ]
    color_map = {
        ' ': 1,
        '#': 2,
        '*': 3,
    }
    def __init__(
        self,
        map_name, # define height, width, wall, lava
        init_loc,
        goal_loc,
        reward_config=dict(
            step=0.0,
            lava=-1.0,
            #wall=0.0,
            goal=1.0,
        ),
    ):
        self.map_name = map_name
        self.map = read_map(Path('maps', '{}.txt'.format(map_name)))
        self.init_loc = init_loc
        self.goal = goal_loc
        self.reward_config = reward_config
        self.observation_space = spaces.Tuple((spaces.Discrete(len(self.map)), spaces.Discrete(len(self.map[0]))))
        self.action_spasce = spaces.Discrete(4)
        self.size = (len(self.map), len(self.map[0]))
        self._render = None

    # this is used for multitask, so that you can change the parameters to get different environments
    # or inverse engineering data
    def get_parameters(self):
        return dict(
            map_name=self.map_name,
            init_loc=self.init_loc,
            goal=self.goal,
            reward_config=self.reward_config,
        )

    def set_parameters(self, params):
        self.map_name = params['map_name']
        self.init_loc = params['init_loc']
        self.goal = params['goal']
        self.reward_config = params['reward_config']

    def is_valid_loc(self, loc): # whether it is a nonterminal state in state space
        return self.map[loc[0]][loc[1]] != '#' and loc != self.goal

    def reset(self):
        self.state = self.init_loc

    def _transition(self, action):
        ds = ds_dict[action]
        next_state = (self.state[0] + ds[0], self.state[1] + ds[1])
        if not self.observation_space.contains(next_state) or self.map[next_state[0]][next_state[1]] == '#':
            next_state = self.state
        done = next_state == self.goal
        return next_state, done

    def _r(self, action, next_state):
        if next_state == self.goal:
            return self.reward_config['goal'] + self.reward_config['step']
        elif self.map[self.state[0]][self.state[1]] == '*':
            return self.reward_config['lava']
        else:
            return self.reward_config['step']

    def step(self, action):
        assert self.state != self.goal, 'env already terminates'
        next_state, done = self._transition(action)
        r = self._r(action, next_state)
        self.state = next_state
        return self.state, r, done, self.get_info()

    def get_info(self):
        return self.get_parameters()

    def render(self, mode='human'):
        if self._render is None:
            self._render = Render()
            self.drawer = GridDrawer(self.__class__.color_list)
        color_map = self.__class__.color_map
        indices = np.zeros(self.size, dtype=int)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (i, j) == self.state:
                    indices[i, j] = 0
                else:
                    indices[i, j] = color_map[self.map[i][j]]
        self._render.render(self.drawer.draw(indices)[:,:,:3])

# puddle
class ContinuousGridWorld(Env):
    pass

# every time you reset, you call a function to set the paraemters of your env
class SampleParameterEnv(Wrapper):
    def __init__(self, env, sample_param_f):
        super().__init__(env)
        self.sample_param_f = sample_param_f # do sampling

    def reset(self):
        self.env.set_parameters(self.sample_param_f(self.env.get_parameters()))
        return self.env.reset()
