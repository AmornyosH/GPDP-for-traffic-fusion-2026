# A Script For Environment's Parameters
import traci
import numpy as np


# Constants
SUMO_TIMESTEP = 0.1

class DoubleRing:
    def __init__(self):
        def _paramsDict():
            return {'environment': 'double_ring',
                    'time_step': SUMO_TIMESTEP,
                    'warmup_time': (500//SUMO_TIMESTEP),
                    'one_hour': (3600//SUMO_TIMESTEP),
                    'horizon': (1000//SUMO_TIMESTEP),
                    'state_dim': 10,
                    'action_dim': 2,
                    'lateral_action': True, 
                    'detect_leading_dist': 100,
                    'max_speed_allowed': 10, 
                    'min_speed_allowed': 0}
        
        # Initialise Environment's Parameters
        self.params_dict = _paramsDict()
        self.state_dim = self.params_dict['state_dim']
        self.action_dim = self.params_dict['action_dim']
        self.lateral_action = True  # Lateral action is allowed in this environment.
        self.detect_leading_dis = self.params_dict['detect_leading_dist']
        self.max_speed_allowed = self.params_dict['max_speed_allowed']
        self.min_speed_allowed = self.params_dict['min_speed_allowed']

        # Initialise SUMO
        ...

    def reset(self):
        # Reset the environment and return the initial state s_0 ~ d(s_0).
        return self.getObservation()
    
    def getObservation(self, veh_id, noise=False):
        # ----- State conditions (Follow Alex Bayen's paper):
        state = np.zeros([self.state_dim], dtype=float)
        state[0] = traci.vehicle.getSpeed(vehID=veh_id)  # Get It's speed (m/s)
        state[1] = traci.vehicle.getLaneIndex(vehID=veh_id)  # Get current lane's index (-)
        # Get Leader on AV's current lane
        leader_info = traci.vehicle.getLeader(vehID=veh_id, dist=self.detect_leading_dis)
        # Check if there is a vehicle in front ?
        if leader_info != None:
            # Yes -- There is
            state[2] = traci.vehicle.getSpeed(vehID=leader_info[0])
            state[3] = leader_info[1]
        else:
            # No -- There is not
            state[2] = self.max_speed_allowed  # Assign the maximum speed (Assume that the leader is faster than AV.)
            state[3] = self.detect_leading_dis  # Assign distance = maximum observeable distance.

        # -------------------------- If the state dimension has more than 3 (double ring and so on.)
        # Get Follower info on AV's current lane
        follower_info = traci.vehicle.getFollower(vehID=veh_id, dist=self.detect_leading_dis)
        # Check if there is a vehicle behind ?
        if follower_info[0] != '':
            # Yes -- There is
            state[4] = traci.vehicle.getSpeed(vehID=follower_info[0])
            state[5] = follower_info[1]
        else:
            # No -- There is not
            state[4] = self.min_speed_allowed  # Assign the minimum speed (Assume that the follower is slower than AV.)
            state[5] = self.detect_leading_dis  # Assign distance = maximum observeable distance.

        # Get Leader and follower of the other lane based on the current lane of AV.
        if state[1] == 0:  # AV is on the outer lane (get left neighbor)
            # Get the leader and follower of the inner lane (left --> mode bit 1: 0)
            leader_mode_set = 0b010
            follower_mode_set = 0b000
        else:# AV is on the inner lane (get right neighbor)
            # Get the leaser and follower of the outer lane (right --> mode bit 1: 1)
            leader_mode_set = 0b011
            follower_mode_set = 0b001
        # Get the leader and follower info
        neighbor_follower_info = np.squeeze(traci.vehicle.getNeighbors(vehID=veh_id, mode=follower_mode_set))
        neighbor_leader_info = np.squeeze(traci.vehicle.getNeighbors(vehID=veh_id, mode=leader_mode_set))
        # Check if there is a neighbor leader ?
        if len(neighbor_leader_info) != 0:
            # Yes -- There is
            state[6] = traci.vehicle.getSpeed(vehID=neighbor_leader_info[0])
            state[7] = neighbor_leader_info[1]
        else:
            # No -- There is not
            state[6] = self.max_speed_allowed  # Assign the maximum speed (Assume that the leader is faster than AV.)
            state[7] = self.detect_leading_dis # Assign distance = maximum observeable distance.
        # Check if there is a neighbor follower ?
        if len(neighbor_follower_info) != 0:
            # Yes -- There is
            state[8] = traci.vehicle.getSpeed(vehID=neighbor_follower_info[0])
            state[9] = neighbor_follower_info[1]
        else:
            # No -- There is not
            state[8] = self.min_speed_allowed  # Assign the minimum speed (Assume that the follower is slower than AV.)
            state[9] = self.detect_leading_dis  # Assign distance = maximum observeable distance.

        # Inject noise to the observation (if noise=True)
        if noise == True:
            noise_factor = 5  # If == 0, no noise in the observation
            for n in range(len(state)):
                if n == 0:
                    # No noise in it's own speed
                    ran_noise = 0
                else:
                    ran_noise = np.random.random() * noise_factor
                state[n] += ran_noise
        # Return state
        return state
    
    def assignActions(self, action):
        # Acceleration
        # Lateral action
        ...

    def step(self, action):
        # Interact with SUMO
        # Return next_observation, reward, terminated, truncrated

        # !!!!!!!!!! Check the action's dimension...
        if len(action) != 2:
            print('The dimension of action is not correct. Please try again.')
            exit()

        self.assignActions(action)  # Assign actions.
        traci.simulationStep()  # Taking a step.

        _next_state = self.getObservation()
        _reward = 0
        _terminated = False
        _truncrated = False
        return _next_state, _reward, _terminated, _truncrated
        ...