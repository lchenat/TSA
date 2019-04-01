import gym
import time
import unittest
import numpy as np

from env import MultiGoalReacherEnv, DiscretizeActionEnv

class TestOrigin(unittest.TestCase):
    def test(self):
        env = gym.make('Reacher-v2')
        env.reset()
        env.render()
        done = False
        while not done:
            _, r, done, _ = env.step(env.action_space.sample())
            env.render()

class TestEnv(unittest.TestCase):
    def test_random(self):
        goals = [
            (0, 0.15),
            (0, -0.15),
            (0.15, 0),
            (-0.15, 0),
        ]
        env = DiscretizeActionEnv(
            MultiGoalReacherEnv(goals),
            (5, 5),
        )
        while True:
            o = env.reset()
            env.render()
            for _ in range(100):
                a = env.action_space.sample()
                #a = np.array([0.0, 0.5])
                print(len(o))
                o, r, done, info = env.step(a)
                print(info)
                #assert np.all(env.get_body_com('target') == np.array(env.goal + (0.01,)))
                env.render()
                time.sleep(0.1)     


if __name__ == "__main__":
    unittest.main()
