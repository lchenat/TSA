from gym import Env, Wrapper

class ParameterizedEnv(Env):
    def get_parameters(self):
        raise NotImplementedError

    def set_parameters(self, params):
        raise NotImplementedError

# save current state in self.state
class MDPEnv(ParameterizedEnv):
    def _transition(self, state, action):
        raise NotImplementedError

    def _r(self, state, action, next_state=None):
        raise NotImplementedError
        
    def get_info(self, state=None, action=None, next_state=None):
        raise NotImplementedError

    def step(self, action):
        next_state, done = self._transition(self.state, action)
        self.state = next_state
        return next_state, self._r(self.state, action, next_state), done, self.get_info(self.state, action, next_state)

class SimulateEnv(ParameterizedEnv):
    def _transition(self, action):
        raise NotImplementedError

    def _r(self, action, next_state=None):
        raise NotImplementedError
        
    def get_info(self, action=None, next_state=None):
        raise NotImplementedError

    def step(self, action):
        next_state, done = self._transition(action)
        self.state = next_state
        return next_state, self._r(action, next_state), done, self.get_info(action, next_state)

# every time you reset, you call a function to set the paraemters of your env
class SampleParameterEnv(Wrapper):
    def __init__(self, env, sample_param_f):
        super().__init__(env)
        self.sample_param_f = sample_param_f # do sampling

    def reset(self):
        self.env.set_parameters(self.sample_param_f(self.env.get_parameters()))
        return self.env.reset()

    def get_parameters(self):
        return self.env.get_parameters()

    def set_parameters(self, params):
        return self.env.set_parameters(params)

# be careful of using this!
class LastWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def __getattr__(self, attr):
        env = self.env
        while True:
            try:
                return getattr(env, attr)
            except:
                if env.unwrapped == env or isinstance(env, LastWrapper): break
                env = env.env 
        raise AttributeError('[LastWrapper] no attribute: {}'.format(attr))
