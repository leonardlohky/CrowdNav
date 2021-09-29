# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist

def estimate_reward_on_predictor(state, next_state):
    """ If the time step is small enough, it's okay to model agent as linear movement during this period
    """
    # collision detection
    if isinstance(state, list) or isinstance(state, tuple):
        state = tensor_to_joint_state(state)
    human_states = state.human_states
    robot_state = state.robot_state

    next_robot_state = next_state.robot_state
    next_human_states = next_state.human_states
    cur_position = np.array((robot_state.px, robot_state.py))
    end_position = np.array((next_robot_state.px, next_robot_state.py))
    goal_position = np.array((robot_state.gx, robot_state.gy))
    reward_goal = 0.01 * (norm(cur_position - goal_position) - norm(end_position - goal_position))
    # check if reaching the goal
    reaching_goal = norm(end_position - np.array([robot_state.gx, robot_state.gy])) < robot_state.radius
    dmin = float('inf')
    collision = False
    collision_penalty = 0
    for i, human in enumerate(human_states):
        next_human = next_human_states[i]
        px = human.px - robot_state.px
        py = human.py - robot_state.py
        ex = next_human.px - next_robot_state.px
        ey = next_human.py - next_robot_state.py
        # closest distance between boundaries of two agents
        closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - robot_state.radius
        if closest_dist < 0:
            collision = True
        elif closest_dist < dmin:
            dmin = closest_dist
        if closest_dist < 0.2:
            collision_penalty = collision_penalty + (closest_dist - 0.2) * 0.25 * 0.5
    reward_col = 0
    if collision:
        reward_col = -0.25
    elif reaching_goal:
        reward_goal = 1 + reward_goal
    reward = reward_col + reward_goal + collision_penalty
    # if collision:
        # reward = reward - 100
    reward = reward * 10
    
    return reward