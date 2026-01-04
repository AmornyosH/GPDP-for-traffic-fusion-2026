import numpy as np
import torch
from time import time

from agents.AV import AutonomousVehicle
from agents.RLs.SGDQ import StochasticGaussianprocessDiffusionQlearning

TAG = '(Training) ----->'
CAUTION_TAG = '(Training) !!!!!>'
PRECISION = 4

# ========== Offline Training
def offlineTraining(path:str, env:object, alg:str, cont:bool, dataset:dict, total_steps:int):
    # Initialise agent's object
    if alg == 'SGDP':
        from agents.RLs.SGDP import StochasticGaussianprocessDiffusionPolicy
        agent = StochasticGaussianprocessDiffusionPolicy(params_dict=env.params_dict, dataset=dataset, load=cont)
    elif alg == 'SGDQ':
        from agents.RLs.SGDQ import StochasticGaussianprocessDiffusionQlearning
        agent = StochasticGaussianprocessDiffusionQlearning(params_dict=env.params_dict, dataset=dataset, load=cont)
    else:
        print(f'{CAUTION_TAG} Please provide validable name of the algorithms. Program is terminated!')
        exit()

    # Initialise parameters
    _steps = agent.training_step_counter
    _total_steps = total_steps
    _epoch = agent.training_record
    _steps_per_epoch = int(dataset['num_sample']/agent.MINIBATCH_SIZE)
    _total_epochs = int(_total_steps/_steps_per_epoch)

    # Start training
    while _epoch < _total_epochs:
        _start_time = time()
        _training_results = agent.training(steps=_steps_per_epoch, online=False)
        _epoch += 1
        print(f'{TAG} Epoch: {_epoch}/{_total_epochs},',
              f' Gradient Step: {agent.training_step_counter}/{_total_steps}',
              ' P-Loss: ', round(_training_results['policy_loss'], PRECISION),
              ' Q1-Loss: ', round(_training_results['q1_loss'], PRECISION),
              ' Q2-Loss: ', round(_training_results['q2_loss'], PRECISION) if _training_results.get('q2_loss') is not None else '',
              ' GP-Loss: ', round(_training_results['gp_loss'], PRECISION) if _training_results.get('gp_loss') is not None else '',
              ' Time: ', round(time()-_start_time, PRECISION)
              )
        
    # GP training.
    if agent.ALG == 'SGDP':
        # For SGDP, the GPR model is trained seperately. 
        agent.gp_model.y_train = agent.getAlteredObservation(states=agent.gp_model.x_train)
        # agent.gp_model.y_train = agent.getAlteredObservation(states=agent.gp_model.x_train_original)
        agent.gp_model.myTraining(total_epoch=256)
        agent.recordSaving(path=agent.training_record_path)

# ========== Online Training
def onlineTraining(path:str, env:object, alg:str, cont:bool, dataset:dict, total_steps:int, imposter:bool=False):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialise vehicle objects
    if alg == 'SGDP':
        agent = AutonomousVehicle(load=True, params_dict=env.params_dict, dataset=dataset, id='my_AV')

    # Initialise MDP transitions and others
    # observations_buffer = []
    # actions_buffer = []
    # rewards_buffer = []
    # next_observation_buffer = []
    # truncrated_buffer = []
    # termination_buffer = []
    horizon = env.params_dict['horizon']
    trajectory_count = 0  # Trajectory counter
    step_count = 0

    # 
    while step_count < total_steps:
        
        # Initialise env.
        _ = env.reset(veh=agent, num_veh=40)
        _truncrated = False
        _termination = False
        _accum_reward = 0

        # Initialise MDP transitions
        observations_buffer = []
        actions_buffer = []
        rewards_buffer = []
        next_observation_buffer = []
        truncrated_buffer = []
        termination_buffer = []

        # Agent observes initial state.
        _observation = agent.getObservation()

        # Initialise Imposter Vehicle
        _impos_veh_id = env.initImposter(av_lane=_observation[1])


        # Interaction Loop
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

            # Storing transitions
            torch.vstack([agent.controller.state_buffer, torch.tensor(_observation.reshape(1, -1))])
            torch.vstack([agent.controller.action_buffer, torch.tensor(_action)])
            torch.vstack([agent.controller.reward_buffer, torch.tensor(_reward)])
            torch.vstack([agent.controller.next_state_buffer, torch.tensor(_next_observation)])
            # truncrated_buffer.append(_truncrated)
            # termination_buffer.append(_termination)

            # Storing transitions
            observations_buffer.append(_observation)
            actions_buffer.append(_action)
            rewards_buffer.append(_reward)
            next_observation_buffer.append(_next_observation)
            truncrated_buffer.append(_truncrated)
            termination_buffer.append(_termination)

            # if len(observations_buffer) >= num_data:
            #     break

            _training_results = agent.controller.training(steps=1, online=True)

            # print(f'{TAG} Epoch: {agent.controller.training_record},',
            #     f' Gradient Step: {agent.controller.training_step_counter}',
            #     ' P-Loss: ', round(_training_results['policy_loss'], 2),
            #     ' Q1-Loss: ', round(_training_results['q1_loss'], 2),
            #     ' Q2-Loss: ', round(_training_results['q2_loss'], 2) if _training_results.get('q2_loss') is not None else '',
            #     ' GP-Loss: ', round(_training_results['gp_loss'], 2) if _training_results.get('gp_loss') is not None else '',
            #     #   ' Time: ', round(time()-_start_time, 2)
            #     )

            if t >= horizon*0.2 and imposter:
                env.setImposterSpeed(id=_impos_veh_id)

            # Assign new state
            _observation = _next_observation
            step_count += 1

        # Count the trajectory
        trajectory_count += 1

        # Close the SUMO connection. (To not show error message)
        env.close()

        # Train GP
        # For SGDP, the GPR model is trained seperately. 
        agent.controller.gp_model.prepareTempData(temp_states=np.vstack(observations_buffer).reshape(-1, 10), 
                                                  temp_actions=np.vstack(actions_buffer).reshape(-1, 2))
        agent.controller.gp_model.y_train = agent.controller.getAlteredObservation(states=agent.controller.gp_model.x_train)
        # agent.controller.gp_model.y_train = agent.controller.getAlteredObservation(states=agent.controller.gp_model.x_train_original)
        agent.controller.gp_model.myTraining(total_epoch=10, online=True)
        # agent.controller.recordSaving(path=agent.controller.training_record_path)

        print(f'{TAG} Epoch: {agent.controller.training_record},',
            f' Gradient Step: {agent.controller.training_step_counter}',
            ' P-Loss: ', round(_training_results['policy_loss'], PRECISION),
            ' Q1-Loss: ', round(_training_results['q1_loss'], PRECISION),
            ' Q2-Loss: ', round(_training_results['q2_loss'], PRECISION) if _training_results.get('q2_loss') is not None else '',
            ' GP-Loss: ', round(_training_results['gp_loss'], PRECISION) if _training_results.get('gp_loss') is not None else '',
            #   ' Time: ', round(time()-_start_time, 2)
            )

        # Print summary status.
        print(f'{env.env_tag}', f'{agent.veh_tag}', 'Trajectory completed!', 
                                                    ', Avg_raw_return: ', round(_accum_reward, PRECISION),
                                                    ', Num_vehicles: ', len(env.veh_id_list), 
                                                    ', Depart_lane: ', _observation[1])
    
    return _accum_reward
    ...