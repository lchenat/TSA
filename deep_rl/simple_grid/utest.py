import random
import unittest
import readchar
from PIL import Image
from env import DiscreteGridWorld, RenderEnv
from exemplar_env import RandomInitEnv, RandomGoalEnv

from py_tools.common import set_seed
from py_tools.common.test import ipdb_on_exception

control_dict={'w': 0, 's': 1, 'a': 2, 'd': 3} # for control

class TestDiscrete(unittest.TestCase):
    @ipdb_on_exception
    def control(self):
        #env = DiscreteGridWorld('fourroom', (1, 1), (9, 9))
        env = RandomGoalEnv(
            RandomInitEnv(
                DiscreteGridWorld('fourroom', (1, 1), (9, 9)),
                min_dis=7,
            ),
            goal_locs=[(1, 1), (1, 9), (9, 1), (9, 9)],
        )
        env = RenderEnv(env)
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
            elif c == 'p': # save the observation
                Image.fromarray(env.get_img()).save('observation.jpg')
            elif c in control_dict:
                a = control_dict[c]
                o, r, done, _ = env.step(a)
                env.render()
                print('s={}, r={}'.format(o, r))


if __name__ == "__main__":
    set_seed(1)
    unittest.main()
