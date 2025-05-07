# ======================================== SUMO Releated =========================================
# import os
# import sys
# if 'SUMO_HOME' in os.environ:
#     sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
# import traci
# from sumolib import checkBinary

import numpy as np

from utilities import my_utilities as my_utils

class IDMAutonomousVehicle:
    def __init__(self, load:bool, params_dict:dict, datasets:dict=None, id='AV'):
        self.load = load
        self.vehicle_id = id
        self.vehicle_type = 'rl_agent'
        self.sumo_time_step = params_dict['time_step']  # sumo time step
        self.env_config = params_dict['environment']
        self.action_config = params_dict['action_config']  # action configuration (discrete or continuous)
        self.reward_objective = params_dict['reward_objective']  # reward objective (greedy or global)
        self.reward_config = params_dict['reward_config']  # reward configuration (original or my_reward)
        self.num_lane = params_dict['num_lane']
        self.num_veh = params_dict['num_veh']
        self.training_flag = params_dict['training_flag']
        self.test_flag = params_dict['test_flag']
        
        # AV's information
        self.lane_id = np.arange(self.num_lane)
        self.state_dim = params_dict['state_dim']
        self.action_dim = params_dict['action_dim']
        self.lateral = params_dict['lateral_action']
        self.max_speed = 10.
        self.min_speed = 0.
        self.lead_dist = 100.
        self.min_accel = -0.5  # it is a 'deceleration' but using the word min accel instead.
        self.max_accel = 0.5
        self.min_action_space = -1.
        self.max_action_space = 1.

        # Other Variables
        self.reward_append = []  # Used in reward function.
        self.accum_reward_append = []  # Used in reward function.
        self.accum_reward = 0  # Used in reward function.

    # Action Encoding Method 
    def actionEncoding(self, *args):
        '''
        Encode the action from SUMO or agent to the action space (In this case is for diffusion models).
        '''
        inputs = list(args)
        encoded_actions = np.zeros([self.action_dim], dtype=float)
        
        encoded_actions[0] = my_utils.rangeConversion(inputs[0], 
                                                      new_min=self.min_action_space,
                                                      new_max=self.max_action_space,
                                                      old_min=-1.0,
                                                      old_max=0.5)
        if self.action_dim > 1:
            if inputs[1] == 0:
                encoded_actions[1] = np.random.normal(loc=0.5, scale=0.05)
            else:
                encoded_actions[1] = np.random.normal(loc=-0.5, scale=0.05)
        else:
            if inputs[1] == 0:
                encoded_actions[0] = encoded_actions[0]
            else:
                encoded_actions[0] = -1.*encoded_actions[0]

        return encoded_actions.tolist()

    # Action Decoding Method
    def actionDecoding(self, inputs):
        decoded_actions = np.zeros([2 if self.lateral else 1], dtype=float)  # Long action + Lat action
        # If the environment has multiple lanes.
        if len(decoded_actions) > 1:
            decoded_actions[0] = my_utils.rangeConversion(inputs[0],
                                                          old_min=self.min_action_space,
                                                          old_max=self.max_action_space,
                                                          new_min=self.min_accel,
                                                          new_max=self.max_accel)
            if inputs[1] > 0.4 and inputs[1] < 0.6:
                decoded_actions[1] = self.lane_id[0]
            elif inputs[1] < -0.4 and inputs[1] > -0.6:
                decoded_actions[1] = self.lane_id[1]
            else:
                decoded_actions[1] = self.getState()[1]

        else:
            decoded_actions = my_utils.rangeConversion(inputs,
                                                       old_min=self.min_action_space,
                                                       old_max=self.max_action_space,
                                                       new_min=self.min_accel,
                                                       new_max=self.max_accel)
        return decoded_actions
    
    # # Action Performing Method
    # def actionPerform(self, state, size:int=1, s_print:bool=False, lane_id:int=None, guide=False, predictor='main'):
    #     action_long = traci.vehicle.getAcceleration(vehID=self.vehicle_id)
    #     action_lat = traci.vehicle.getLaneIndex(vehID=self.vehicle_id) if lane_id is None else lane_id
    #     action_sample = self.actionEncoding(action_long, action_lat)
    #     # Print the select actions
    #     if s_print:
    #         print('action_sample: ', action_sample,
    #               ', action_long: ', round(action_long, 2), 
    #               ', action_lat: ', action_lat if self.lateral else 'None', 
    #               ', av_speed: ', round(state[0], 2),
    #               ', av_lane: ', state[1])
    #     duration = int(1/self.sumo_time_step)  # 1s/sumo_time_step(0.1s)   
    #     # Perform lateral action
    #     if self.lateral:
    #         traci.vehicle.changeLane(vehID=self.vehicle_id, laneIndex=action_lat, duration=duration)
    #     return action_sample

    # # Reward Function Method
    # def rewardFunction(self, n_veh:int, noise:bool=False):
    #     # Define hyperparameters
    #     _sum_speed = 0  # Reset sum_speed (For global objective)
    #     _num_convolute = 5  # Convolution kernel size (1-D) (Used for computing running mean/std)
    #     _gamma = 0.99
    #     # Get next state first
    #     _next_state = self.getState(noise)
        
    #     # Check for the reward configuration,
    #     # ------ Alex Bayen's Paper ------
    #     if self.reward_config == 'original':
    #         # Derive Avg Speed
    #         for c in range(n_veh-1):
    #             _sum_speed += traci.vehicle.getSpeed(vehID=str(c))
    #         avg_speed = (_sum_speed + _next_state[0]) / n_veh
    #         # Check reward objective
    #         if self.reward_objective == 'greedy':
    #             # Using the greedy objective since it might be feasible in real-world. (Just my assumption)
    #             # Reward = AV's speed (Greedy objective)
    #             reward = _next_state[0]  
    #         elif self.reward_objective == 'global': 
    #             # Reward = All vehicles average speed (Global objective)
    #             reward = avg_speed
    #         # Append Reward
    #         self.reward_append.append(reward)

    #         # Append Accumalated Reward
    #         self.accum_reward = (_gamma * self.accum_reward) + reward
    #         self.accum_reward_append.append(self.accum_reward)

    #         # Compute running mean/std
    #         if len(self.reward_append) <= _num_convolute*2:
    #             mean_t = np.mean(self.reward_append[0:len(self.reward_append)])
    #             std_t = np.std(self.accum_reward_append[0:len(self.reward_append)])
    #         else:
    #             m = len(self.reward_append)-_num_convolute//2
    #             mean_t = np.mean(self.reward_append[0+m:m+_num_convolute])
    #             std_t = np.std(self.accum_reward_append[0+m:_num_convolute+m])
            
    #         # Normalise variance of reward 
    #         norm_reward = (reward - mean_t) / (std_t + 1e-08)

    #     # ------ My Modified Reward Fn ------
    #     elif self.reward_config == 'my_reward':
    #         # ------ Greed Objective -------
    #         if self.reward_objective == 'greedy':
    #             reward = _next_state[0]  # Get AV's speed.
    #         # # ------ Global Objective ------
    #         elif self.reward_objective == 'global':
    #             # reward = avg_speed
    #             reward = (_next_state[0]+_next_state[2]+_next_state[4]+_next_state[6]+_next_state[8])/5
    #         # Append Reward
    #         self.reward_append = np.append(self.reward_append, reward)
    #         # Derive running mean
    #         kernel_pos = len(self.reward_append) - _num_convolute
    #         if kernel_pos <= _num_convolute:
    #             mean_t = np.mean(self.reward_append[0:_num_convolute])
    #         else:
    #             mean_t = np.mean(self.reward_append[0+kernel_pos : _num_convolute+kernel_pos])
    #         # Get distance penalty
    #         # This is the penalty when the av is too far behind from the leading vehicle.
    #         # dis_penalty = [0.2, 1]  ('1' means that the av is in the perfect clearance.)
    #         # dis_penalty = 1 - ((next_state[3] - self.min_gap)/self.lead_dist)
    #         dis_penalty = 1  # Try fix value first (No distance penalty).
    #         # Normalised the reward
    #         norm_reward = (mean_t / self.max_speed) * dis_penalty
    #     else:
    #         print('Unknown reward configuration.({:s}) Please try again.'.format(self.reward_config))
    #         exit()  # Terminate the script.

    #     # Return tuples
    #     return norm_reward, _next_state

    # Storages Reset Method
    def resetStorages(self):
        self.reward_append = []
        self.accum_reward_append = []
        self.accum_reward = 0
        







