from deep_rl.gridworld import ReachGridWorld, PORGBEnv, ReachGoalManager
from ipdb import slaunch_ipdb_on_exception
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['cluster'], default='cluster')
parser.add_argument('--n_goal', type=int, default=10)
parser.add_argument('--min_dis', type=int, default=3)
parser.add_argument('--n_abs', type=int, default=10)
args = parser.parse_args()

colors = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

class one_hot:
    # 1 or 2 dim
    @staticmethod
    def encode(indices, dim):
        if len(indices.shape) > 1:
            indices = indices.squeeze(axis=1)
        assert len(indices.shape) == 1, 'shape error'
        return np.eye(dim)[indices]

    # 2-dim
    @staticmethod
    def decode(vs):
        return np.argmax(vs, 1)

def cluster_abstraction(env, n_abs):
    from sklearn.cluster import KMeans
    n_action = env.action_space.n
    env.reset()
    states = [s for s in env.pos_candidates if s not in goals]
    m = [[] for _ in range(len(states))]
    for combo in env.train_combos:
        env.reset(index=combo)
        for i, s in enumerate(states):
            env.teleport(*s)
            qs = np.array(env.get_q(0.99))
            best_actions = (qs == qs.max()).astype(int)
            #m[i].append(one_hot.encode(np.random.choice(best_actions, size=1), n_action)[0])
            m[i].append(best_actions)
    m = np.array([np.concatenate(row) for row in m])
    kmeans = KMeans(n_clusters=n_abs, random_state=0).fit(m)
    return {s: label for s, label in zip(states, kmeans.labels_)}
    

def visualize(abstract_map, env):
    m = env.get_map(0)
    for i in range(len(m)):
        for j in range(len(m[i])):
            if m[i][j] == '#': continue
            elif (i, j) in abstract_map:
                m[i][j] = str(abstract_map[(i, j)])
            else:
                m[i][j] = 'G'
    for row in m:
        print(' '.join(row))
                

if __name__ == '__main__':
    with slaunch_ipdb_on_exception():
        map_names = ['map49']
        goal_manager = ReachGoalManager(map_names[0])
        # min_dis between goals (approximation of corelation)
        goals = goal_manager.gen_goals(args.n_goal + 1, min_dis=args.min_dis) 
        train_combos = [(0,) + goal for goal in goals[:args.n_goal]]
        test_combos = [(0,) + goal for goal in goals[args.n_goal:]]
        env = ReachGridWorld(map_names, train_combos, test_combos)
        if args.method == 'cluster':
            abstract_map = cluster_abstraction(env, args.n_abs)
        # abstract_map {(x, y): abstract state index}
        visualize(abstract_map, env)
