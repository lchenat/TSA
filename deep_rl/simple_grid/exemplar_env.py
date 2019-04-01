import random
if __package__ == '':
    from base_env import SampleParameterEnv
    from env import DiscreteGridWorld
else:
    from .base_env import SampleParameterEnv
    from .env import DiscreteGridWorld, SampleParameterEnv


class RandomInitEnv(SampleParameterEnv):
    def __init__(self, env, min_dis=1):
        self.min_dis = min_dis
        def sample_param_f(param):
            while True:
                loc = (random.randint(1, self.env.size[0]-2), random.randint(0, self.env.size[1]-2))
                if self.env.is_valid_loc(loc) and self.env.dist2goal(loc) >= self.min_dis: break
            param['init_loc'] = loc
            return param
        super().__init__(env, sample_param_f)

class RandomGoalEnv(SampleParameterEnv):
    def __init__(self, env, goal_locs):
        self.goals = goal_locs
        def sample_param_f(param):
            self.goal_idx = random.randint(0, len(self.goals)-1)
            param['goal'] = self.goals[self.goal_idx] 
            return param
        super().__init__(env, sample_param_f)
