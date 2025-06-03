# ======================================== SUMO Releated =========================================
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
from sumolib import checkBinary

import numpy as np
import torch
import random

from utilities import miscellaneous as misc
from agents.RLs.SGDP import StochasticGaussianprocessDiffusionPolicy


class AutonomousVehicle:
    def __init__(self, load:bool, params_dict:dict, dataset:dict=None, id='AV'):
        self.load = load
        self.veh_id = id
        self.veh_type = 'my_AV'
        self.veh_tag = '(myAV) ----->'
        self.veh_caution_tag = '(myAV) !!!!!>'
        self.sumo_time_step = params_dict['time_step']  # sumo time step
        self.env_name = params_dict['environment']
        self.num_lane = params_dict['num_lane']
        self.num_veh = params_dict['num_veh']
        self.reward_config = params_dict['reward_config']
        self.reward_objective = params_dict['reward_objective']
        
        # AV's information
        self.lane_id = np.arange(self.num_lane)
        self.observation_dim = params_dict['observation_dim']
        self.action_dim = params_dict['action_dim']
        self.lateral_action = params_dict['lateral_action']
        self.max_speed = 10.
        self.min_speed = 0.
        self.detect_dist = 100.
        self.min_accel = -1.5  # It is a 'deceleration' but using the word min accel instead.
        self.max_accel = 1.0
        self.min_action_space = -1.
        self.max_action_space = 1.

        # Other Variables
        self.reward_append = []  # Used in reward function.
        self.accum_reward_append = []  # Used in reward function.
        self.accum_reward = 0  # Used in reward function.

        # Initialise High Level control algorithm 
        self.controller = StochasticGaussianprocessDiffusionPolicy(params_dict=params_dict, dataset=dataset)

    # Observation Method
    def getObservation(self, noise=False):
        # ----- State conditions (Follow Alex Bayen's paper):
        observation = np.zeros([self.observation_dim], dtype=float)
        observation[0] = traci.vehicle.getSpeed(vehID=self.veh_id)  # Get It's speed (m/s)
        observation[1] = traci.vehicle.getLaneIndex(vehID=self.veh_id)  # Get current lane's index (-)
        # Get Leader on AV's current lane
        leader_info = traci.vehicle.getLeader(vehID=self.veh_id, dist=self.detect_dist)
        # Check if there is a vehicle in front ?
        if leader_info != None:
            # Yes -- There is
            observation[2] = traci.vehicle.getSpeed(vehID=leader_info[0])
            observation[3] = leader_info[1]
        else:
            # No -- There is not
            observation[2] = self.max_speed  # Assign the maximum speed (Assume that the leader is faster than AV.)
            observation[3] = self.detect_dist  # Assign distance = maximum observeable distance.

        # -------------------------- If the state dimension has more than 3 (double ring and so on.)
        # Get Follower info on AV's current lane
        follower_info = traci.vehicle.getFollower(vehID=self.veh_id, dist=self.detect_dist)
        # Check if there is a vehicle behind ?
        if follower_info[0] != '':
            # Yes -- There is
            observation[4] = traci.vehicle.getSpeed(vehID=follower_info[0])
            observation[5] = follower_info[1]
        else:
            # No -- There is not
            observation[4] = self.max_speed  # Assign the minimum speed (Assume that the follower is slower than AV.)
            observation[5] = self.detect_dist  # Assign distance = maximum observeable distance.

        # Get Leader and follower of the other lane based on the current lane of AV.
        if observation[1] == 0:  # AV is on the outer lane (get left neighbor)
            # Get the leader and follower of the inner lane (left --> mode bit 1: 0)
            leader_mode_set = 0b010
            follower_mode_set = 0b000
        else:# AV is on the inner lane (get right neighbor)
            # Get the leaser and follower of the outer lane (right --> mode bit 1: 1)
            leader_mode_set = 0b011
            follower_mode_set = 0b001
        # Get the leader and follower info
        neighbor_follower_info = np.squeeze(traci.vehicle.getNeighbors(vehID=self.veh_id, mode=follower_mode_set))
        neighbor_leader_info = np.squeeze(traci.vehicle.getNeighbors(vehID=self.veh_id, mode=leader_mode_set))
        # Check if there is a neighbor leader ?
        if len(neighbor_leader_info) != 0:
            # Yes -- There is
            observation[6] = traci.vehicle.getSpeed(vehID=neighbor_leader_info[0])
            observation[7] = neighbor_leader_info[1]
        else:
            # No -- There is not
            observation[6] = self.max_speed  # Assign the maximum speed (Assume that the leader is faster than AV.)
            observation[7] = self.detect_dist # Assign distance = maximum observeable distance.
        # Check if there is a neighbor follower ?
        if len(neighbor_follower_info) != 0:
            # Yes -- There is
            observation[8] = traci.vehicle.getSpeed(vehID=neighbor_follower_info[0])
            observation[9] = neighbor_follower_info[1]
        else:
            # No -- There is not
            observation[8] = self.max_speed  # Assign the minimum speed (Assume that the follower is slower than AV.)
            observation[9] = self.detect_dist  # Assign distance = maximum observeable distance.

        # Inject noise to the observation (if noise=True)
        if noise == True:
            noise_factor = 5  # If == 0, no noise in the observation
            for n in range(len(observation)):
                if n == 0:
                    # No noise in it's own speed
                    ran_noise = 0
                else:
                    ran_noise = np.random.random() * noise_factor
                observation[n] += ran_noise
        return observation

    # Action Encoding Method 
    def actionEncoding(self, *args):
        '''
        Encode the action from SUMO or agent to the action space (In this case is for diffusion models).
        '''
        inputs = list(args)
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        encoded_actions = torch.zeros([self.action_dim], dtype=torch.float32)
        encoded_actions[0] = misc.rangeConversion(inputs[0], 
                                                      new_min=self.min_action_space,
                                                      new_max=self.max_action_space,
                                                      old_min=self.min_accel,
                                                      old_max=self.max_accel)
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
        decoded_actions = np.zeros([2 if self.lateral_action else 1], dtype=float)  # Long action + Lat action
        # If the environment has multiple lanes.
        if len(decoded_actions) > 1:
            decoded_actions[0] = misc.rangeConversion(inputs[0],
                                                          old_min=self.min_action_space,
                                                          old_max=self.max_action_space,
                                                          new_min=self.min_accel,
                                                          new_max=self.max_accel)
            if inputs[1] > 0.4 and inputs[1] < 0.6:
                decoded_actions[1] = self.lane_id[0]
            elif inputs[1] < -0.4 and inputs[1] > -0.6:
                decoded_actions[1] = self.lane_id[1]
            else:
                decoded_actions[1] = self.getObservation()[1]

        else:
            decoded_actions = misc.rangeConversion(inputs,
                                                       old_min=self.min_action_space,
                                                       old_max=self.max_action_space,
                                                       new_min=self.min_accel,
                                                       new_max=self.max_accel)
        return decoded_actions
    
    # Action Performing Method
    def actionPerform(self, state, s_print:bool=False, lane_id:int=None):
        # Random agent will random actions both longitudinal and lateral.
        _pred = self.controller.predict(state=state, size=1, guide=True)
        _pred = self.actionDecoding(_pred.tolist()[0])
        print(_pred)
        action_long = _pred[0]
        action_lat = _pred[1]
        # print('Random acc: ', action_long, ', Random lane: ', action_lat)
        action_sample = self.actionEncoding(action_long, action_lat)
        # Print the select actions
        if s_print:
            print('action_sample: ', action_sample,
                  ', action_long: ', round(action_long, 2), 
                  ', action_lat: ', action_lat if self.lateral_action else 'None', 
                  ', av_speed: ', round(state[0], 2),
                  ', av_lane: ', state[1])
        duration = int(1/self.sumo_time_step)  # 1s/sumo_time_step(0.1s)   
        # Perform lateral action
        if self.lateral_action:
            traci.vehicle.changeLane(vehID=self.veh_id, laneIndex=action_lat, duration=duration)
        return action_sample

    # Next Observation and Raw reward Method
    def getNextObsAndReward(self, noise:bool=False, scaled:bool=True):
        # Get next state first
        _next_state = self.getObservation(noise)

        # Check for the reward configuration,
        if self.reward_objective == 'greedy':
            # Greedy objective <-- self speed
            _reward = _next_state[0]  # AV's speed
        elif self.reward_objective == 'global':
            # Global objective <-- overall speed in the env.

            # # Follow Cathy Wu's paper.
            # # Derive Avg Speed of all vehicles in the env. 
            # # This config may not be feasible since the problem is formulated in POMDP. 
            # for c in range(self.num_veh-1):
            #     _sum_speed += traci.vehicle.getSpeed(vehID=str(c))
            # _avg_speed = (_sum_speed + _next_state[0]) / self.num_veh
            # # Reward = All vehicles average speed (Global objective)
            # _reward = _avg_speed

            # My global reward configuration
            # We can only observe the overall speed from our observation. 
            # Which contains only 5 vehicles, including self speed.
            _reward = (_next_state[0]+_next_state[2]+_next_state[4]+_next_state[6]+_next_state[8])/5
        
        # If the reward needs to be scaled... (Default = True)
        if scaled:
            _reward /= self.max_speed

        return _next_state, _reward

    # Normalising Reward Function Method
    def normRewardFunction(self, noise:bool=False):
        # Storages Reset Method
        def _resetStorages():
            self.reward_append = []
            self.accum_reward_append = []
            self.accum_reward = 0

        # Define hyperparameters
        _sum_speed = 0  # Reset sum_speed (For global objective)
        _num_convolute = 4  # Convolution kernel size (1-D) (Used for computing running mean/std)
        _gamma = 0.99
        _resetStorages()

        # Get next state first
        _next_state = self.getObservation(noise)
        
        # Check for the reward configuration,
        # ------ Alex Bayen's Paper ------
        if self.reward_config == 'original':
            # Check reward objective
            if self.reward_objective == 'greedy':
                # Using the greedy objective since it might be feasible in real-world. (Just my assumption)
                # Reward = AV's speed (Greedy objective)
                reward = _next_state[0]  
            elif self.reward_objective == 'global': 
                # Derive Avg Speed
                for c in range(self.num_veh-1):
                    _sum_speed += traci.vehicle.getSpeed(vehID=str(c))
                avg_speed = (_sum_speed + _next_state[0]) / self.num_veh
                # Reward = All vehicles average speed (Global objective)
                reward = avg_speed
            # Append Reward
            self.reward_append.append(reward)

            # Append Accumalated Reward
            self.accum_reward = (_gamma * self.accum_reward) + reward
            self.accum_reward_append.append(self.accum_reward)

            # Compute running mean/std
            m = len(self.reward_append)
            mean_t = np.mean(self.reward_append[0+np.abs(m-_num_convolute//2):m+_num_convolute//2])
            std_t = np.std(self.accum_reward_append[0+np.abs(m-_num_convolute//2):m+_num_convolute//2])
            
            # Normalise variance of reward 
            norm_reward = (reward - mean_t) / (std_t + 1e-08)

        # ------ My Modified Reward Fn ------
        elif self.reward_config == 'my_reward':
            # ------ Greed Objective -------
            if self.reward_objective == 'greedy':
                reward = _next_state[0]  # Get AV's speed.
            # # ------ Global Objective ------
            elif self.reward_objective == 'global':
                # reward = avg_speed
                reward = (_next_state[0]+_next_state[2]+_next_state[4]+_next_state[6]+_next_state[8])/5
            # Append Reward
            self.reward_append = np.append(self.reward_append, reward)
            # Derive running mean
            kernel_pos = len(self.reward_append) - _num_convolute
            if kernel_pos <= _num_convolute:
                mean_t = np.mean(self.reward_append[0:_num_convolute])
            else:
                mean_t = np.mean(self.reward_append[0+kernel_pos : _num_convolute+kernel_pos])
            # Get distance penalty
            # This is the penalty when the av is too far behind from the leading vehicle.
            # dis_penalty = [0.2, 1]  ('1' means that the av is in the perfect clearance.)
            # dis_penalty = 1 - ((next_state[3] - self.min_gap)/self.lead_dist)
            dis_penalty = 1  # Try fix value first (No distance penalty).
            # Normalised the reward
            norm_reward = (mean_t / self.max_speed) * dis_penalty
        else:
            print('Unknown reward configuration.({:s}) Please try again.'.format(self.reward_config))
            exit()  # Terminate the script.

        # Return tuples
        return _next_state, norm_reward






