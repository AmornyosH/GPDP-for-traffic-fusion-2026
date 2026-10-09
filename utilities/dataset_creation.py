from agents.IDM import IDM
from agents.RandomAV import RandomAV
from agents.DerivedAV import DerivedAV

import random
import torch
import numpy as np


def datasetCreation(env: object, num_data: int, type: str = 'expert'):
    """
    Collect offline dataset D = {o_n, a_n}^{N-1}_{n=0} of MDP transitions.

    Paper Section V (Training Phase):
      'expert'      — DerivedAV on outer lane (lane 0, optimal) for the first half,
                      then DerivedAV on inner lane (lane 1, sub-optimal) for the second
                      half.  Mixing optimal and sub-optimal trajectories from the same
                      behaviour policy follows the dataset construction described in the
                      paper.
      'human'       — IDM only (human-driving baseline).
      'human-expert'— DerivedAV (outer lane) for first half, IDM for second half.
      'human-random'— IDM for first half, RandomAV for second half.

    Each MDP transition stored: (o_t, a_t, r_{t+1}, o_{t+1}, truncated, terminated).
    Vehicle density cycles through env.num_veh_config across trajectories so that
    the dataset covers the full evaluation range [40, 42, 44, 46, 48] vehicles.

    Args:
        env       : DoubleRing environment object.
        num_data  : Total number of transitions to collect (paper uses 10^6).
        type      : Dataset type — see above.

    Returns:
        dict with keys: num_sample, observations, actions, rewards,
                        next_observations, truncrated, terminations.
    """
    horizon = env.params_dict['horizon']

    # MDP transition buffers
    obs_buf       = []
    act_buf       = []
    rew_buf       = []
    next_obs_buf  = []
    truncated_buf = []
    term_buf      = []

    trajectory_count = 0

    while len(obs_buf) < num_data:

        # ------------------------------------------------------------------ #
        # Select behaviour agent and depart lane for this trajectory.         #
        # Paper: first half optimal (outer lane), second half sub-optimal     #
        # (inner lane), both using the derived policy (Section V).            #
        # ------------------------------------------------------------------ #
        half = num_data // 2
        collected = len(obs_buf)

        if type == 'expert':
            agent = DerivedAV(load=False, params_dict=env.paramsDict(),
                              dataset=None, id='my_Derived')
            # First half: outer lane (lane=0) → higher speed → optimal actions.
            # Second half: inner lane (lane=1) → lower speed → sub-optimal actions.
            lane = 0 if collected < half else 1

        elif type == 'human':
            agent = IDM(load=False, params_dict=env.paramsDict(),
                        dataset=None, id='my_IDM')
            lane = None

        elif type == 'human-expert':
            if collected < half:
                agent = DerivedAV(load=False, params_dict=env.paramsDict(),
                                  dataset=None, id='my_Derived')
                lane = 0
            else:
                agent = IDM(load=False, params_dict=env.paramsDict(),
                            dataset=None, id='my_IDM')
                lane = None

        elif type == 'human-random':
            if collected < half:
                agent = IDM(load=False, params_dict=env.paramsDict(),
                            dataset=None, id='my_IDM')
            else:
                agent = RandomAV(load=False, params_dict=env.paramsDict(),
                                 dataset=None, id='my_Random')
            lane = None

        else:
            raise ValueError(
                f"Unknown dataset type '{type}'. "
                "Choose from ['expert', 'human', 'human-expert', 'human-random']."
            )

        # ------------------------------------------------------------------ #
        # Cycle vehicle density across num_veh_config so the dataset covers   #
        # all evaluation densities: [40, 42, 44, 46, 48] veh.                #
        # ------------------------------------------------------------------ #
        num_veh = env.num_veh_config[trajectory_count % len(env.num_veh_config)]

        # Reset: adds all vehicles, runs warmup (500 / δ steps) so congestion
        # forms before data collection begins (follows paper Section V).
        env.reset(veh=agent, num_veh=num_veh, lane=lane)

        accum_reward = 0.0
        obs = agent.getObservation()   # initial observation o_0

        # ------------------------------------------------------------------ #
        # Collect one trajectory of MDP transitions                           #
        # (o_t, a_t, r_{t+1}, o_{t+1}).                                      #
        # ------------------------------------------------------------------ #
        for t in range(horizon):
            # Behaviour policy selects and applies action a_t.
            # lane_id pins the agent to the designated lane for this trajectory.
            action = agent.actionPerform(state=obs, lane_id=lane)

            # Step environment → o_{t+1}, r_{t+1}
            next_obs, reward, terminated, truncated = env.step(veh=agent)
            accum_reward += reward

            # Mark the final transition of the trajectory.
            if t + 1 >= horizon:
                truncated  = True
                terminated = True

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            next_obs_buf.append(next_obs)
            truncated_buf.append(truncated)
            term_buf.append(terminated)

            if len(obs_buf) >= num_data:
                break

            obs = next_obs

        trajectory_count += 1
        env.close()

        print(
            f'{env.env_tag} {agent.veh_tag}'
            f'  traj {trajectory_count:4d}'
            f' | num_veh: {num_veh:3d}'
            f' | lane: {lane}'
            f' | accum_reward: {accum_reward:8.2f}'
            f' | collected: {len(obs_buf):7d} / {num_data}'
        )

    # ---------------------------------------------------------------------- #
    # Assemble dataset dictionary.                                            #
    # Keys match what GPDP, miscellaneous.py, and the GP models expect.      #
    # ---------------------------------------------------------------------- #
    obs_dim = env.params_dict['observation_dim']
    act_dim = env.params_dict['action_dim']

    dataset = {
        'num_sample':        num_data,
        'observations':      np.vstack(obs_buf[:num_data]).reshape(num_data, obs_dim),
        'actions':           np.vstack(act_buf[:num_data]).reshape(num_data, act_dim),
        'rewards':           np.array(rew_buf[:num_data], dtype=np.float32).reshape(num_data, 1),
        'next_observations': np.vstack(next_obs_buf[:num_data]).reshape(num_data, obs_dim),
        'truncrated':        np.array(truncated_buf[:num_data]),   # key kept as-is (used downstream)
        'terminations':      np.array(term_buf[:num_data]),
    }

    print('\n--- Dataset Summary ---')
    print(f"  type         : {type}")
    print(f"  num_sample   : {dataset['num_sample']}")
    print(f"  observations : {dataset['observations'].shape}")
    print(f"  actions      : {dataset['actions'].shape}")
    print(f"  rewards      : {dataset['rewards'].shape}")
    print(f"  next_obs     : {dataset['next_observations'].shape}")
    print(f"  truncrated   : {dataset['truncrated'].shape}")
    print(f"  terminations : {dataset['terminations'].shape}")

    return dataset
