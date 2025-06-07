import numpy as np
import torch

from agents.IDM import IDM
from agents.RandomAV import RandomAV
from agents.AV import AutonomousVehicle

def sumoInteraction(env:object, num_veh:int, alg:str, dataset:dict):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialise vehicle objects
    if alg == 'SGDP':
        agent = AutonomousVehicle(load=False, params_dict=env.params_dict, dataset=dataset, id='my_AV')

    # Initialise MDP transitions and others
    observations_buffer = []
    actions_buffer = []
    rewards_buffer = []
    next_observation_buffer = []
    truncrated_buffer = []
    termination_buffer = []
    horizon = env.params_dict['horizon']
    trajectory_count = 0  # Trajectory counter

    # Initialise env.
    _ = env.reset(veh=agent, num_veh=num_veh)
    _truncrated = False
    _termination = False
    _accum_reward = 0

    # Agent observation initial state
    _observation = agent.getObservation()

    # Loop
    for t in range(horizon):
        # Agent take action
        _action = agent.actionPerform(state=_observation, s_print=False)

        # Take a step in the environment and observes termination. 
        _next_observation, _reward, _truncrated, _termination = env.step(veh=agent)
        _accum_reward += _reward
        # Check if this is the final timestep.
        if t+1 >= horizon:
            _truncrated = True
            _termination = True # Set termination true just for now.

        # # Storing transitions
        # observations_buffer.append(_observation)
        # actions_buffer.append(_action)
        # rewards_buffer.append(_reward)
        # next_observation_buffer.append(_next_observation)
        # truncrated_buffer.append(_truncrated)
        # termination_buffer.append(_termination)

        # if len(observations_buffer) >= num_data:
        #     break

        # Assign new state
        _observation = _next_observation

    # Count the trajectory
    trajectory_count += 1

    # Close the SUMO connection. (To not show error message)
    env.close()

    # Print summary status.
    print(f'{env.env_tag}', f'{agent.veh_tag}', 'Trajectory completed!', 
                                                ', Avg_raw_return: ', round(_accum_reward, 2),
                                                ', Num_vehicles: ', len(env.veh_id_list), 
                                                ', Depart_lane: ', _observation[1])
    
    return _accum_reward