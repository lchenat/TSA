from gym import Env

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
        return next_state, self._r(action, next_state), done, self.get_info(action, next_state)

