from agents.IDM import IDM
import numpy as np

def datasetCreation(env:object, num_data:int):
    horizon = env.params_dict['horizon']
    # Initialise IDM vehicle
    idm_agent = IDM(load=False, params_dict=env.params_dict, datasets=None, id='my_IDM')

    # Initialise MDP transitions
    observations_buffer = []
    actions_buffer = []
    rewards_buffer = []
    next_observation_buffer = []
    truncrated_buffer = []
    termination_buffer = []

    # Start
    while len(observations_buffer) < num_data:
        # Initialise env.
        _ = env.reset(veh=idm_agent)
        _truncrated = False
        _termination = False
        _accum_reward = 0

        # Agent observation initial state
        _observation = idm_agent.getObservation()

        # Loop
        for t in range(horizon):
            # Agent take action
            _action = idm_agent.actionPerform(state=_observation)

            # Take a step in the environment and observes termination. 
            _next_observation, _reward, _truncrated, _termination = env.step(veh=idm_agent)
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
        
        # Close the SUMO connection. (To not show error message)
        env.close()

        # Print summary status.
        print(f'{env.env_tag}', f'{idm_agent.veh_tag}', 'Trajectory completed!', 
                                                        ', Avg_raw_reward: ', round(_accum_reward, 2),
                                                        ', Num_vehicles: ', len(env.veh_id_list), 
                                                        ', Depart_lane: ', _observation[1],
                                                        ', Dataset_size: ', len(observations_buffer))
        

    # Contruct dataset dictionary.
    _dataset = {'observations': np.vstack(observations_buffer).reshape(num_data, env.params_dict['observation_dim']), 
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