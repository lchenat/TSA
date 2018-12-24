from deep_rl import *
from deep_rl.gridworld import ReachGridWorld, PORGBEnv, ReachGoalManager
from ipdb import slaunch_ipdb_on_exception
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['cluster'], default='cluster')
parser.add_argument('--n_goal', type=int, default=10)
parser.parse_args('--min_dis', type=int, default=3)
parser.parse_args('--n_abs', type=int, default=10)

def cluster_abstraction(env, n_abs):
    states = 
    env.reset()
    for 

def visualize(map_id, abstract_map, env)
    pass

if __name__ == '__main__':
    map_names = ['map49']
    goal_manager = ReachGoalManager(map_names[0])
    # min_dis between goals (approximation of corelation)
    goals = goal_manager.gen_goals(args.n_goal + 1, min_dis=args.min_dis) 
    train_combos = [(0,) + goal for goal in goals[:args.n_goal]]
    test_combos = [(0,) + goal for goal in goals[args.n_goal:]]
    env = ReachGridWorld(map_names, train_combos, test_combos)
    if args.method == 'cluster':
        abstract_map = cluster_abstraction(env, config.n_abs)
    # abstract_map {(x, y): abstract state index}
    visualize(map_id, abstract_map, env)
