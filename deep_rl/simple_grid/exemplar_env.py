import random
if __package__ == '':
    from env import DiscreteGridWorld, SampleParameterEnv
else:
    from .env import DiscreteGridWorld, SampleParameterEnv


class RandomInitDiscreteGridWorld(SampleParameterEnv):
    def __init__(
        self,
        map_name,
        goal_loc,
        reward_config=dict(
            step=0.0,
            lava=-1.0,
            #wall=0.0,
            goal=1.0,
        ),
        discount=0.99,
        seed=0, # does not use here
    ):
        env = DiscreteGridWorld(map_name, init_loc=(0, 0), goal_loc=goal_loc, reward_config=reward_config, discount=discount, seed=seed)
        def sample_param_f(param):
            while True:
                loc = (random.randint(1, self.env.size[0]-2), random.randint(0, self.env.size[1]-2))
                if self.env.is_valid_loc(loc): break
            param['init_loc'] = loc
            return param
        super().__init__(env, sample_param_f)
