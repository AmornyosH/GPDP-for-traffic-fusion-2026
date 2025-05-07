# A Script For Environment's Parameters
import numpy as np
# ======================================== SUMO Releated =========================================
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
from sumolib import checkBinary

# Constants
SUMO_TIMESTEP = 0.1

class DoubleRing:
    def __init__(self, num_veh:int=44, render:bool=False):
        def _paramsDict():
            return {'environment': 'double_ring',
                    'simulation_config': 'sumo' if not render else 'sumo-gui',
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
        self.env_name = self.params_dict['environment']
        self.env_tag = '(Double_Ring) ---------->'
        self.caution_tag = '(Double_Ring) !!!!!!!!!!>'
        self.state_dim = self.params_dict['state_dim']
        self.action_dim = self.params_dict['action_dim']
        self.lateral_action = True  # Lateral action is allowed in this environment.
        self.detect_leading_dis = self.params_dict['detect_leading_dist']
        self.max_speed_allowed = self.params_dict['max_speed_allowed']
        self.min_speed_allowed = self.params_dict['min_speed_allowed']
        self.num_veh = num_veh  # Number of vehicles (default = 44)

        # Initialise SUMO
        # ========== Initial command for opening the sumo interface.
        self.sumo_binary = checkBinary(self.params_dict['simulation_config'])
        self.sumo_cmd = [self.sumo_binary, "-c", "environments/{:s}/{:s}.sumocfg.xml".format(self.env_name, self.env_name), 
                        '--no-warnings', 
                        '--no-step-log']
        # Note: minGap has been changed to 2.0 (Not default value 2.5)
        # Note: sigma of "rl_agent" has been set to 0 (Not default value 0.2) <-- which means the AV's driving is perfect.
        # ========== Pre-define simulation environment
        self.route_id = 0  # Get all route's ID -- There is only 1 route in this environment (right to left)
        self.veh_type = ['IDM', 'my_AV']  # Get the vehicle type
        self.lane_id = ['left_0', 'left_1', 'right_0', 'right_1']  # Get all lane ids in the environment.
        # Print the status of the environments
        print(f'{self.env_tag}', 'Environment Initialised !')
        print(f'{self.env_tag}', 'Route_id: ', self.route_id)
        print(f'{self.env_tag}', 'Veh_type: ', self.veh_type)
        print(f'{self.env_tag}', 'Lane_id: ', self.lane_id)
        
    def reset(self, veh_id:str=None):
        '''
        Add the AV to the environment (SUMO) and return initial observation (s_0). \\
        Arguments: Vehicle's ID (veh_id) \\
        Returns: Initial observation (o_0), current_timestep (step)
        '''
        # Reset the environment and return the initial state s_0 ~ d(s_0). 
        self.start()
        _step = 0  # Initialise step counter

        # !!!!!!!!!! Check if the vehicle's id is provided ?
        if veh_id is None:
            print(f'{self.caution_tag}', 'The vehicles\'s state cannot be observed. Please provide vehicle id !')
            exit()
        # `````````` Generate Autonomous Vehicle in the simulation (Agent)
        traci.vehicle.add(vehID=veh_id, routeID=self.route_id, typeID='my_AV', departLane=0)
        traci.simulationStep() 
        _step += 1

        # =================================== Run-in steps start ===================================
        # Run-in steps until all vehicles have been deployed.
        # Generate Non-Controlled Vehicles in the simulation
        for i in range(self.num_veh-1):
            traci.vehicle.add(vehID=i, routeID=self.route_id, typeID='IDM', departLane="allowed")  # Add vehicle into the simulation in another lane
            traci.vehicle.setSpeed(vehID=i, speed=0.5)  # Slow every vehicle in the run-in steps
            traci.vehicle.setLaneChangeMode(i, 0b000000000000)  # Prohibit lane change to the IDM vehicles
        traci.simulationStep()  # Take 1st simulation step to prevent below condition to be ignored.
        # Limit the speed of agent to wait for the other vehicles to depart.
        while len(traci.simulation.getPendingVehicles()) != 0:
            # >>>>>>>>>> In run-in state, AV speed is limited to slow (In order to wait for all the vehicle to depart)
            _run_in_speed = 0.5
            traci.vehicle.setSpeed(vehID=veh_id, speed=_run_in_speed)
            # >>>>>>>>>> Step Increment
            traci.simulationStep()  # Simulation Step
            _step += 1 # Assigned check point of time for run-in time.

        # `````````` After all vehicle have been departed, set the speed control back to SUMO configuration.
        for i in range(self.num_veh-1):
            traci.vehicle.setSpeed(vehID=i, speed=-1)
        traci.vehicle.setSpeed(vehID=veh_id, speed=-1)
        traci.simulationStep()
        _step += 1
        _run_in_time = _step
        # ==================================== Run-in steps end ====================================

        # =================================== Warmup steps start ===================================
        # `````````` Warmup Steps
        # Warm up the traffic to get a steady state of heavy traffic before learning (Follow the paper.)
        while _step < (_run_in_time + self.params_dict['warmup_time']):
            traci.simulationStep()
            _step += 1
        # ==================================== Warmup steps end ====================================

        # Return the initial state
        return self.getObservation(veh_id=veh_id)

    def start(self):
        # !!!!!!!!!! Check if the SUMO is connected ?
        if traci.isLoaded():
            self.close()  # Close an active connection. 
        traci.start(self.sumo_cmd)

    def close(self):
        traci.close()

    def getObservation(self, veh_id:str, noise=False):
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