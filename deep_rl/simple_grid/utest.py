import unittest
import readchar
from env import DiscreteGridWorld
from utils import Render, GridDrawer

from py_tools.common import set_seed
from py_tools.common.test import ipdb_on_exception

class TestDiscrete(unittest.TestCase):
    @ipdb_on_exception
    def control(self):
        env = DiscreteGridWorld('maps/fourroom.txt', (1, 1), (9, 9))
        env.reset()
        env.render()
        done = False
        while not done:
            c = readchar.readchar()
            if c  == 'q':
                break
            elif c == 'r':
                env.reset()
                env.render()
            elif c in env.__class__.control_dict:
                a = env.__class__.control_dict[c]
                _, r, done, _ = env.step(a)
                env.render()
                print('s={}, r={}'.format(env.state, r))

if __name__ == "__main__":
    set_seed(1)
    unittest.main()
