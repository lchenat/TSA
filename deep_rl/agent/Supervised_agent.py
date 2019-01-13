

class SupervisedBaseAgent:
    def __init__(self, config):
        self.config = config

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

class SupervisedAgent(SupervisedBaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network)
        self.total_steps = 0
 
    def step(self):
        config = self.config
        states, infos = config.eval_env.env.envs[0].last.get_teleportable_states(config.discount)
        states = tensor(states)
        infos = stack_dict(infos)
        probs = self.network.get_probs(states, infos)
        labels = tensor(infos['opt_a'], dtype=torch.long)
        self.opt.step()
