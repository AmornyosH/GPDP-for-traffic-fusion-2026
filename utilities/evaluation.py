import numpy as np
import torch
import random

from agents.IDM import IDM
from agents.RandomAV import RandomAV
from agents.AV import AutonomousVehicle
from agents.DerivedAV import DerivedAV

def sumoInteraction(env:object, num_veh:int, alg:str, dataset:dict, load:bool, imposter:bool=False, eval_time:int=1):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialise vehicle objects
    if alg == 'SGDP':
        agent = AutonomousVehicle(load=load, params_dict=env.params_dict, dataset=dataset, id='my_AV', online=False)
    elif alg == 'derived':
        agent = DerivedAV(load=False, params_dict=env.paramsDict(), dataset=None, id='my_Derived')
    elif alg == 'IDM':
        agent = IDM(load=False, params_dict=env.paramsDict(), dataset=None, id='my_Derived')

    # Initialise MDP transitions and others
    observations_buffer = []
    actions_buffer = []
    rewards_buffer = []
    next_observation_buffer = []
    truncrated_buffer = []
    termination_buffer = []
    horizon = env.params_dict['horizon']
    trajectory_count = 0  # Trajectory counter

    avg_av_speed_append = []
    avg_all_speed_append = []
    accum_reward_append = []

    for i in range(eval_time):

        # ========== Set global seed generator first. 
        seed = 2203 + (i * 100)
        random.seed(seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed)

        # Initialise env.
        # _ = env.reset(veh=agent, num_veh=num_veh, imposter=imposter)
        _ = env.reset(veh=agent, num_veh=num_veh, imposter=imposter, lane=1)
        _truncrated = False
        _termination = False
        _accum_reward = 0

        # Agent observation initial state
        _observation = agent.getObservation()

        # Initialise Imposter Vehicle
        _impos_veh_id = env.initImposter(av_lane=_observation[1])

        # Loop
        for t in range(horizon):
            # Agent take action
            if t % (1) == 0:
                _action = agent.actionPerform(state=_observation, s_print=False, pred=True)
            else:
                _action = agent.actionPerform(state=_observation, s_print=False, pred=False)
            # _action = agent.actionPerform(state=_observation, s_print=True)

            # Take a step in the environment and observes termination. 
            _next_observation, _reward, _truncrated, _termination = env.step(veh=agent)
            _accum_reward += _reward
            # Check if this is the final timestep.
            if t+1 >= horizon:
                _truncrated = True
                _termination = True # Set termination true just for now.

            # Get speed info of vehs...
            avg_av_speed_append.append(_next_observation[0])
            avg_all_speed_append.append(env.getSumSpeed(num_veh=num_veh)/num_veh)

            # Uncertainty triggering
            if t >= horizon*0.2 and imposter:
                env.setImposterSpeed(id=_impos_veh_id)
                # if t % (horizon*0.2) == 0:
                #     _impos_veh_id = env.initImposter(av_lane=_observation[1])
                #     print("changed!")

            # Assign new state
            _observation = _next_observation

        # Count the trajectory
        trajectory_count += 1

        # Close the SUMO connection. (To not show error message)
        env.close()

        # Append accumulative reward
        accum_reward_append.append(_accum_reward)

        # Print summary status.
        print(f'{env.env_tag}', f'{agent.veh_tag}', 'Trajectory completed!', 
                                                    ', Avg_raw_return: ', round(_accum_reward, 2),
                                                    ', Num_vehicles: ', len(env.veh_id_list), 
                                                    ', Depart_lane: ', _observation[1])
        
    eval_results = {'mean_accum_reward': np.mean(accum_reward_append), 
                    'std_accum_reward': np.std(accum_reward_append), 
                    'mean_av_speed': np.mean(np.vstack(avg_av_speed_append)), 
                    'std_av_speed': np.std(np.vstack(avg_av_speed_append)), 
                    'mean_all_speed': np.mean(np.vstack(avg_all_speed_append)), 
                    'std_all_speed': np.std(np.vstack(avg_all_speed_append))}
    
    return eval_results