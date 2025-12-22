from agents.IDM import IDM
from agents.RandomAV import RandomAV
from agents.DerivedAV import DerivedAV

import random
import torch
import numpy as np

def datasetCreation(env:object, num_data:int):
    horizon = env.params_dict['horizon']
    # # Initialise vehicle's objects
    # idm_agent = IDM(load=False, params_dict=env.params_dict, dataset=None, id='my_IDM')
    # random_agent = RandomAV(load=False, params_dict=env.params_dict, dataset=None, id='my_Random')

    # Initialise MDP transitions
    observations_buffer = []
    actions_buffer = []
    rewards_buffer = []
    next_observation_buffer = []
    truncrated_buffer = []
    termination_buffer = []
    trajectory_count = 0  # Trajectory counter
    # current_agent = 'IDM' # Default current agent is IDM.

    # Start
    while len(observations_buffer) < num_data:
        # Select the agent
        idm_agent = IDM(load=False, params_dict=env.paramsDict(), dataset=None, id='my_IDM')
        # random_agent = RandomAV(load=False, params_dict=env.paramsDict(), dataset=None, id='my_Random')
        derived_agent = DerivedAV(load=False, params_dict=env.paramsDict(), dataset=None, id='my_Derived')

        # We need to select the agent before reset the env. 
        if len(observations_buffer) < (num_data // 2):
            # Start with IDM
            # agent = idm_agent
            agent = derived_agent
        else:
            # agent = random_agent
            agent = idm_agent
            # agent = derived_agent

        # Initialise env.
        _num_veh = env.num_veh_config[int(trajectory_count % len(env.num_veh_config))]  # define number of vehicles
        agent.target_speed = (env.circumference - (env.veh_gap_dist*_num_veh/2)) / _num_veh
        print(agent.target_speed)
        _ = env.reset(veh=agent, num_veh=_num_veh)
        _truncrated = False
        _termination = False
        _accum_reward = 0

        # Agent observation initial state
        _observation = agent.getObservation()

        # Loop
        for t in range(horizon):
            # Agent take action
            _action = agent.actionPerform(state=_observation) 
            _action[0] += (0.05 * np.random.random())
            # Take a step in the environment and observes termination. 
            _next_observation, _reward, _truncrated, _termination = env.step(veh=agent)
            _accum_reward += _reward
            # Check if this is the final timestep.
            if t+1 >= horizon:
                _truncrated = True
                _termination = True # Set termination true just for now.

            # Storing transitions
            observations_buffer.append(_observation)
            actions_buffer.append(_action)
            rewards_buffer.append(_reward)
            next_observation_buffer.append(_next_observation)
            truncrated_buffer.append(_truncrated)
            termination_buffer.append(_termination)

            if len(observations_buffer) >= num_data:
                break

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
                                                    ', Depart_lane: ', _observation[1],
                                                    ', Dataset_size: ', len(observations_buffer))
        
    # Contruct dataset dictionary.
    _dataset = {'num_sample': len(observations_buffer),
                'observations': np.vstack(observations_buffer).reshape(num_data, env.params_dict['observation_dim']), 
                'actions': np.vstack(actions_buffer).reshape(num_data, env.params_dict['action_dim']), 
                'rewards': np.vstack(rewards_buffer).reshape(num_data, 1),
                'next_observations': np.vstack(next_observation_buffer).reshape(num_data, env.params_dict['observation_dim']), 
                'truncrated': np.vstack(truncrated_buffer).reshape(num_data), 
                'terminations': np.vstack(termination_buffer).reshape(num_data)}

    # Check shape of all transitions
    print('Observations_buffer_shape: ', _dataset['observations'].shape)
    print('Actions_buffer_shape: ', _dataset['actions'].shape)
    print('Reward_buffer_shape: ', _dataset['rewards'].shape)
    print('Next_observation_buffer_shape: ', _dataset['next_observations'].shape)
    print('Truncrated_buffer_shape: ', _dataset['truncrated'].shape)
    print('Termination_buffer_shape: ', _dataset['terminations'].shape)

    return _dataset
    ...