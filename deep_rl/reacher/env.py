import random
import numpy as np
from gym.envs.mujoco.reacher import ReacherEnv
from gym import Wrapper


# Multigoal, discretize actions (what is the action range?)
class MultiGoalReacherEnv(ReacherEnv):
    def __init__(self, goals):
        self.goals = goals
        super().__init__()

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            #self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            self.goal = random.sample(self.goals, 1)[0] # could add randomness
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        dists = (self.get_body_com('fingertip')[:2] - np.array(self.goals)).reshape(-1)
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            dists,
            #self.get_body_com('fingertip') - self.get_body_com('target'),
        ])

class DiscretizeActionEnv(Wrapper):
    def __init__(self, env, nums):
        pass
