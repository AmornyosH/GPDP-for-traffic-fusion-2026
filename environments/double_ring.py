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
    def __init__(self, render:bool=False, num_veh:int=44):
        # Initialise Environment's Parameters
        self.env_name = 'double_ring'
        self.env_tag = '(Double_Ring) ----->'
        self.caution_tag = '(Double_Ring) !!!!!>'
        self.render = render
        self.max_num_veh = num_veh
        self.num_veh_config = [40, 42, 44, 46, 48]
        # self.num_veh_config = [33, 40, 44, 49, 58]
        # self.num_veh_config = [44]
        self.veh_id_list = []  # Vehicle's id list.
        self.circumference = 250.
        self.veh_gap_dist = 2.
    
        # Initialise SUMO
        # ========== Initial command for opening the sumo interface.
        self.sumo_binary = checkBinary('sumo' if not render else 'sumo-gui')
        self.sumo_cmd = [self.sumo_binary, "-c", "environments/{:s}/{:s}.sumocfg.xml".format(self.env_name, self.env_name), 
                        '--no-warnings', 
                        '--no-step-log']
        # Note: minGap has been changed to 2.0 (Not default value 2.5)
        # Note: sigma of "rl_agent" has been set to 0 (Not default value 0.2) <-- which means the AV's driving is perfect.
        # ========== Pre-define simulation environment
        self.route_id = 0  # Get all route's ID -- There is only 1 route in this environment (right to left)
        self.veh_type = ['IDM', 'my_AV']  # Get the vehicle type
        # self.lane_id = ['left_0', 'left_1', 'right_0', 'right_1']  # Get all lane ids in the environment.
        self.lane_id = [0, 1]  # Get all lane ids in the environment. 0=outer lane, 1=inner lane
        # Print the status of the environments
        print(f'{self.env_tag}', 'Environment Initialised !')
        print(f'{self.env_tag}', 'Route_id: ', self.route_id)
        print(f'{self.env_tag}', 'Veh_type: ', self.veh_type)
        print(f'{self.env_tag}', 'Lane_id: ', self.lane_id)

        self.params_dict = self.paramsDict()
        self.prev_depart_lane_index = self.params_dict['default_depart_lane_index']

        self.impos_veh_id = None

    def paramsDict(self):
            return {'environment': self.env_name,
                    'time_step': SUMO_TIMESTEP,
                    'warmup_time': int(500/SUMO_TIMESTEP),
                    'one_hour': int(3600/SUMO_TIMESTEP),
                    'horizon': int(1000/SUMO_TIMESTEP),
                    'circumference': self.circumference,
                    'state_dim': 20,
                    'observation_dim': 10,
                    'action_dim': 2,
                    # 'action_dim': 1,
                    'lateral_action': True, 
                    'detect_leading_dist': 100,
                    'veh_gap_dist': self.veh_gap_dist,
                    'max_speed_allowed': 10, 
                    'min_speed_allowed': 0, 
                    'reward_config': 'original', 
                    'reward_objective': 'greedy',
                    'num_lane': 2, 
                    'default_depart_lane_index': 0,
                    'num_veh': self.max_num_veh, 
                    }

    def reset(self, veh:object, num_veh:int=None, imposter:bool=False, lane:int=None):
        '''
        Add the AV to the environment (SUMO) and perform run-in steps, which contain adding 
        all vehicles to the environment, and warming up, which allows the congestion to occur. 
        This method will return initial observation of given vehicle (o_0). 
        
        Arguments : Vehicle's ID (veh_id) 
        Returns : Initial observation (o_0)
        '''
        # Reset the environment and return the initial state s_0 ~ d(s_0). 
        self.start()
        _step = 0  # Initialise step counter
        self.veh_id_list = []  # Reset veh_id list.
        
        # !!!!!!!!!! Check whether the vehicle's id is provided ?
        if veh.veh_id is None:
            print(f'{self.caution_tag}', 'The vehicles\'s state cannot be observed. Please provide vehicle id !')
            exit()
        # `````````` Generate Autonomous Vehicle in the simulation (Agent)
        if lane is None:
            _depart_lane = 0 if self.prev_depart_lane_index != 0 else 1
        else:
            _depart_lane = lane
            # _depart_lane = 1
        self.prev_depart_lane_index = _depart_lane
        traci.vehicle.add(vehID=veh.veh_id, routeID=self.route_id, typeID=veh.veh_type, departLane=_depart_lane)
        self.veh_id_list.append(veh.veh_id)
        traci.simulationStep() 
        _step += 1

        # =================================== Run-in steps start ===================================
        # Run-in steps until all vehicles have been deployed.
        # Determine number of vehicle
        if num_veh is None:
            _total_num_veh = np.random.choice(self.num_veh_config)  # Using random to ensure that all algorithms will experience the same conditions.
        else:
            _total_num_veh = num_veh

        # Generate Non-Controlled Vehicles in the simulation
        for i in range(_total_num_veh-1):
            traci.vehicle.add(vehID=i, routeID=self.route_id, typeID='IDM', departLane="allowed")  # Add vehicle into the simulation in another lane
            traci.vehicle.setImperfection(typeID=i, imperfection=np.random.random()*1)
            self.veh_id_list.append(str(i))  # Store veh_id into the list.
            traci.vehicle.setSpeed(vehID=i, speed=0.5)  # Slow every vehicle in the run-in steps
            traci.vehicle.setLaneChangeMode(i, 0b000000000000)  # Prohibit lane change to the IDM vehicles
        traci.simulationStep()  # Take 1st simulation step to prevent below condition to be ignored.

        # Limit the speed of agent to wait for the other vehicles to depart.
        while len(traci.simulation.getPendingVehicles()) != 0:
            # >>>>>>>>>> In run-in state, AV speed is limited to slow (In order to wait for all the vehicle to depart)
            _run_in_speed = 0.5
            traci.vehicle.setSpeed(vehID=veh.veh_id, speed=_run_in_speed)
            # >>>>>>>>>> Step Increment
            traci.simulationStep()  # Simulation Step
            _step += 1 # Assigned check point of time for run-in time.

        # `````````` After all vehicle have been departed, set the speed control back to SUMO configuration.
        for i in range(_total_num_veh-1):
            traci.vehicle.setSpeed(vehID=i, speed=-1)
        traci.vehicle.setSpeed(vehID=veh.veh_id, speed=-1)  # AV
        traci.simulationStep()
        _step += 1
        _run_in_time = _step
        # ==================================== Run-in steps end ====================================

        # =================================== Warmup steps start ===================================
        # `````````` Warmup Steps
        # Warm up the traffic to get a steady state of heavy traffic before learning (Follow the paper.)
        while _step < (_run_in_time + self.params_dict['warmup_time']):
            # # Add noise to the vehicles.
            # for i in range(_total_num_veh - 1):
            #     _accel = traci.vehicle.getAcceleration(vehID=str(i))
            #     traci.vehicle.setAcceleration(vehID=i, acceleration=_accel+np.random.normal(0, 0.2), duration=self.params_dict['time_step'])
            # traci.vehicle.setSpeed(vehID=veh.veh_id, speed=np.random.normal(0, 0.2))  # AV
            # traci.vehicle.setSpeed(vehID=veh.veh_id, speed=5.25)  # AV
            _obs = veh.getObservation()
            veh.actionPerform(state=_obs, s_print=False, pred=True)
            traci.simulationStep()
            _step += 1
        # ==================================== Warmup steps end ====================================
        traci.vehicle.setSpeed(vehID=veh.veh_id, speed=-1)  # AV

        # Return the initial state
        return _step

    def start(self):
        # !!!!!!!!!! Check if the SUMO is connected ?
        if traci.isLoaded():
            self.close()  # Close an active connection. 
        traci.start(self.sumo_cmd)

    def close(self):
        traci.close()

    def getState(self, noise=False):
        # The state of the environment is a vector of speed of all vehicles.
        _state = []
        for i in self.veh_id_list:
            _state.append(traci.vehicle.getSpeed(vehID=str(i)))
        return _state

    def getSpeed(self, _num_veh):
        _sum_speed = 0
        # Generate Non-Controlled Vehicles in the simulation
        for i in range(_num_veh-1):
            _sum_speed += traci.vehicle.getSpeed(vehID=str(i))
        
        return _sum_speed

    # --------- Initialise imposter vehicle parameters (If noise_inj == True)
    def initImposter(self, av_lane):
        traci.vehicle.setSpeed(vehID=self.impos_veh_id, speed=-1) if self.impos_veh_id is not None else ...
        _lane_id = traci.lane.getIDList()
        _av_lane = int(av_lane)
        _impos_lane = _lane_id[_av_lane]  # Gather current lane of AV. (0=outer, 1=inner)
        # _impos_lane = np.random.choice(_lane_id)
        _veh_in_lane = traci.lane.getLastStepVehicleIDs(laneID=_impos_lane)
        while _veh_in_lane == ():  # Check if there is any vehicle on the current lane of AV
            _impos_lane = np.random.choice(_lane_id)
            _veh_in_lane = traci.lane.getLastStepVehicleIDs(laneID=_impos_lane)
        # --------- Retrieve the imposter vehicle id.
        _impos_veh_id = np.random.choice(traci.lane.getLastStepVehicleIDs(laneID=_impos_lane))
        while _impos_veh_id == 'AV':
            _impos_veh_id = np.random.choice(traci.lane.getLastStepVehicleIDs(laneID=_impos_lane))
        impos_veh_speed = -1  # Initialise imposter's speed with SUMO controlled first.
        self.impos_veh_id = _impos_veh_id

        return _impos_veh_id

    def setImposterSpeed(self, id):
        impos_veh_speed = 1.5  # Slow the imposter vehicle down.
        traci.vehicle.setSpeed(vehID=id, speed=impos_veh_speed)  
        # traci.vehicle.slowDown(vehID=id, speed=impos_veh_speed, duration=SUMO_TIMESTEP) 

    def assignActions(self, action):
        # Acceleration
        # Lateral action
        ...

    def step(self, veh:object):
        # Interact with SUMO
        # Return next_observation, reward, terminated, truncrated

        # self.assignActions(action)  # Assign actions.
        traci.simulationStep()  # Taking a step.

        _next_state, _reward = veh.getNextObsAndReward()
        _terminated = False
        _truncrated = False

        # # ++++++ Noise imposter process (Omitted if noise_imp == False) ++++++
        # # If noise_inj == True, Randomly reduce the speed or stop one of the vehicle 
        # if step > ((_steady_state_time + params_dict['horizon']) * 0.2) and IMPOSTER:
        #     impos_lane = lane_id[int(state[1])]  # Gather current lane of AV. (0=outer, 1=inner)
        #     if traci.lane.getLastStepVehicleIDs(laneID=impos_lane) == ():  # Check if there is any vehicle on the current lane of AV
        #         # If there is no vehicle on the current lane of AV, there must be vehicle on the opposite site (left-right).
        #         if state[1] == 0:  # If AV is currently on the outer lane.
        #             impos_lane = lane_id[2]  # check imposter lane on 
        #         elif state[1] == 1:
        #             impos_lane = lane_id[3]
        #     # traci.lane.getLastStepVehicleIDs(laneID=agent.state[1])
        #     prev_impos_veh_id = impos_veh_id
        #     traci.vehicle.setSpeed(vehID=prev_impos_veh_id, speed=-1)  # Reset current imposter speed (back to SUMO)
        #     # impos_veh_id = np.random.choice(traci.lane.getLastStepVehicleIDs(laneID=impos_lane))  # Generate new imposter vehicle id.
        #     impos_veh_speed = 0.5  # Slow the imposter vehicle down.
        #     traci.vehicle.setSpeed(vehID=impos_veh_id, speed=impos_veh_speed)  

        return _next_state, _reward, _terminated, _truncrated
        ...
