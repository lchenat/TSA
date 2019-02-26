import random
import unittest
import readchar
from env import DiscreteGridWorld, SampleParameterEnv
from utils import Render, GridDrawer

from py_tools.common import set_seed
from py_tools.common.test import ipdb_on_exception

control_dict={'w': 0, 's': 1, 'a': 2, 'd': 3} # for control

class TestDiscrete(unittest.TestCase):
    @ipdb_on_exception
    def control(self):
        env = DiscreteGridWorld('fourroom', (1, 1), (9, 9))
        def sample_param_f(param):
            while True:
                loc = (random.randint(1, env.env.size[0]-2), random.randint(0, env.env.size[1]-2))
                if env.env.is_valid_loc(loc): break
            param['init_loc'] = loc
            return param
        env = SampleParameterEnv(env, sample_param_f)
        o = env.reset()
        env.render()
        done = False
        while not done:
            c = readchar.readchar()
            if c  == 'q':
                break
            elif c == 'r':
                o = env.reset()
                env.render()
            elif c in control_dict:
                a = control_dict[c]
                o, r, done, _ = env.step(a)
                env.render()
                print('s={}, r={}'.format(o, r))

if __name__ == "__main__":
    set_seed(1)
    unittest.main()
