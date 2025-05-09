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
        self.env_tag = '(Double_Ring) ---------->'
        self.caution_tag = '(Double_Ring) !!!!!!!!!!>'
        self.render = render
        self.num_veh = num_veh

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

    def paramsDict(self):
            return {'environment': self.env_name,
                    'time_step': SUMO_TIMESTEP,
                    'warmup_time': int(500/SUMO_TIMESTEP),
                    'one_hour': int(3600/SUMO_TIMESTEP),
                    'horizon': int(1000/SUMO_TIMESTEP),
                    'state_dim': 20,
                    'observation_dim': 10,
                    'action_dim': 2,
                    'lateral_action': True, 
                    'detect_leading_dist': 100,
                    'max_speed_allowed': 10, 
                    'min_speed_allowed': 0, 
                    'reward_config': 'original', 
                    'reward_objective': 'greedy',
                    'num_lane': 2, 
                    'num_veh': self.num_veh, 
                    }

    def reset(self, veh:object):
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

        # !!!!!!!!!! Check whether the vehicle's id is provided ?
        if veh.veh_id is None:
            print(f'{self.caution_tag}', 'The vehicles\'s state cannot be observed. Please provide vehicle id !')
            exit()
        # `````````` Generate Autonomous Vehicle in the simulation (Agent)
        traci.vehicle.add(vehID=veh.veh_id, routeID=self.route_id, typeID=veh.veh_type, departLane=0)
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
            traci.vehicle.setSpeed(vehID=veh.veh_id, speed=_run_in_speed)
            # >>>>>>>>>> Step Increment
            traci.simulationStep()  # Simulation Step
            _step += 1 # Assigned check point of time for run-in time.

        # `````````` After all vehicle have been departed, set the speed control back to SUMO configuration.
        for i in range(self.num_veh-1):
            traci.vehicle.setSpeed(vehID=i, speed=-1)
        traci.vehicle.setSpeed(vehID=veh.veh_id, speed=-1)
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
        return _step

    def start(self):
        # !!!!!!!!!! Check if the SUMO is connected ?
        if traci.isLoaded():
            self.close()  # Close an active connection. 
        traci.start(self.sumo_cmd)

    def close(self):
        traci.close()

    def getEnvState(self, noise=False):
        ...

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
        return _next_state, _reward, _terminated, _truncrated
        ...
