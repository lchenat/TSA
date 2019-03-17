import random
import numpy as np
from gym.envs.mujoco.reacher import ReacherEnv
from gym import Wrapper, spaces


# Multigoal, discretize actions (what is the action range?)
class MultiGoalReacherEnv(ReacherEnv):
    def __init__(self, goals, sample_indices=None, with_goal_pos=True, sparse=False):
        self.goals = goals
        if sample_indices is None:
            sample_indices = list(range(len(goals)))
        self.sample_indices = sample_indices
        self.with_goal_pos = with_goal_pos
        self.sparse = sparse
        super().__init__()
        if with_goal_pos:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8+2*len(self.goals),))
        else:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,))

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        if self.sparse:
            norm = np.linalg.norm(vec)
            reward_dist = 0.05 - norm if norm < 0.05 else 0.0
        else:
            reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict()

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            #self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            self.goal_idx = random.sample(self.sample_indices, 1)[0]
            self.goal = self.goals[self.goal_idx]
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        if self.with_goal_pos:
            dists = (self.get_body_com('fingertip')[:2] - np.array(self.goals)).reshape(-1)
            obs = np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                dists,
            ])
        else:
            obs = np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
            ])
        return obs

    def get_info(self):
        return dict()
        #return dict(reward_dist=0.0, reward_ctrl=0.0)

class DiscretizeActionEnv(Wrapper):
    def __init__(self, env, n_bins):
        super().__init__(env)
        self.n_bins = np.asarray(n_bins)
        self.action_space = spaces.MultiDiscrete(n_bins)
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high

    def step(self, action):
        if isinstance(action, (int, np.int64)): action = (action // self.n_bins[1], action % self.n_bins[1])
        actual_action = self.low + (self.high - self.low) * action / (self.n_bins - 1)
        return self.env.step(actual_action)
