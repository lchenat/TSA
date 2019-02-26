# Design Principle:
- each base env is an MDP, which should be fully specified: discount, transition, reward, etc. Multitask env should be made as an wrapper.
- gym is simulation based environment, therefore reward and transition can only be calculated for current state.
- following simple\_rl, each environment has parameters, which can be tuned during the multitask setting.

# Why not combine with GridWorld?
- The origin gridworld is multitask itself, and have too fancy observations
- The optimal of this new MDP is not shortest path due to the existence of lava (also the preprocessing of map). Therefore the basic setting is different.

# TODO:
- separate render to a wrapper?
