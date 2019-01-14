from ..network import *
from ..component import *

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

    def eval_episodes(self):
        acc = []
        for _ in range(self.config.eval_episodes):
            acc.append(self.eval_episode())
        mean_acc = np.mean(acc)
        self.config.logger.info('eval acc: {}'.format(mean_acc))
        self.config.logger.add_scalar(tag='acc', value=mean_acc, step=self.total_steps)
        return mean_acc

    def eval_episode(self):
        raise NotImplementedError

class SupervisedAgent(SupervisedBaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        if config.label == 'abs':
            self.abs_network = config.abs_network_fn()
        self.opt = config.optimizer_fn(self.network)
        self.total_steps = 0
 
    def eval_episode(self):
        config = self.config
        states, infos = config.eval_env.env.envs[0].last.get_teleportable_states(config.discount)
        states = tensor(states)
        infos = stack_dict(infos)
        if config.label == 'action':
            probs = self.network.get_probs(states, infos)
            labels = tensor(infos['opt_a'], dtype=torch.long)
        elif config.label == 'abs':
            probs = self.network.abs_encoder(states, infos)
            labels = self.abs_network(states, infos)          
        else:
            raise Exception('unsupported label')
        pred_labels = probs.argmax(dim=1)
        return (pred_labels == labels).float().mean() 

    def step(self):
        config = self.config
        states, infos = config.eval_env.env.envs[0].last.get_teleportable_states(config.discount)
        states = tensor(states)
        infos = stack_dict(infos)
        if config.label == 'action':
            probs = self.network.get_probs(states, infos)
            labels = one_hot.encode(tensor(infos['opt_a'], dtype=torch.long), config.action_dim)
        elif config.label == 'abs':
            probs = self.network.abs_encoder(states, infos)
            labels = self.abs_network(states, infos)
        else:
            raise Exception('unsupported label')
        loss_dict = dict()
        loss_dict['NLL'] = (-torch.log(probs) * labels).sum(dim=1).mean()
        loss_dict['network'] = self.network.loss()
        # log before update
        self.loss = loss_dict['NLL'].detach().cpu().numpy()
        for k, v in loss_dict.items():
            config.logger.add_scalar(tag=k, value=v, step=self.total_steps)
        if hasattr(self.network, 'abs_encoder'):
            indices = self.network.abs_encoder.get_indices(states, infos).detach().cpu().numpy()
            abs_map = {pos: i for pos, i in zip(infos['pos'], indices)}
            config.logger.add_file('abs_map', abs_map, step=self.total_steps)
        self.opt.step(sum(loss_dict.values(), 0.0))
        self.total_steps += 1
