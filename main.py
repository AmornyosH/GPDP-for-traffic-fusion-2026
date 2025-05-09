import numpy as np
from time import time, sleep

# ========================================= My Modules ===========================================
# from GPDP_v1.GPDP_v1 import GaussianProcessDiffusionPolicy
# from GPDQ.GPDQ import GaussianProcessDiffusionQlearning
# from SGDP.SGDP import StochasticGaussianprocessDiffusionPolicy
# from Diffusion_QL.Diffusion_QL import DiffusionQL
# # from SAC.sac import SAC
from environments.double_ring import DoubleRing
from dataset_creation import datasetCreation
from agents.IDM import IDM


# ======================================== Other Modules =========================================
import os
import argparse
import random
import torch
CUDA = torch.cuda.is_available()
import matplotlib.pyplot as plt

# ======================================= Local Functions ========================================
def addArguments(parser):
    parser.add_argument('--env', default='simple_sine', help='Environment: ["simple_sine", "Walker2d-v5"], default:"simple_sine"')
    parser.add_argument('--density', default=44, help='The density of the environment (number of vehicle), default:44')
    parser.add_argument('--alg', default="SGDP", help='Algorithm: ["SAC-S", "SAC-D", "D-QL", "GPDP"], default:GPDP')
    parser.add_argument('--task', default='training', help='Task: ["training", "testing"], default:training')
    parser.add_argument('--gradient_step', default=100000, help='Gradient Step, default:1000000')
    parser.add_argument('--eval_mode', default='standard', help='Evaluation mode: [standard, shifted], default:standard')
    parser.add_argument('--rendering', default='no-render', help='Rendering: [render, no-render], default:no-render')

def setGlobalSeed(seed:int):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

# ====================================== Main Program Start ======================================
seed = 2203
setGlobalSeed(seed)
parser = argparse.ArgumentParser(description='SGDP IEEE_Trans_NNLS PyTorch Args')
addArguments(parser)
args = parser.parse_args()

# ========== Global constants
ENV_NAME = args.env
NUM_VEH = int(args.density)
TASK = args.task
EXPERT_PATH = 'datasets/dataset_{}.npz'.format(ENV_NAME)
UNCERTAINTY = True if args.eval_mode == 'shifted' else False
RENDERING = True if args.rendering == 'render' else False
av_id = 'AV_1'

# ========== Create environment object
env = DoubleRing(num_veh=NUM_VEH, render=RENDERING)

# ========== Create IDM agent object
idm_agent = IDM(load=False, params_dict=env.params_dict, datasets=None, id='my_IDM')

# ========== Check whether the dataset is exists ?
dataset_path = 'C:/Amornyos/PhD/resources/datasets/{}_dataset'.format(ENV_NAME)
if os.path.isfile(dataset_path):
    # Load the dataset
    dataset = np.load(dataset_path)
else:
    # Create a new dataset
    dataset = datasetCreation(env=env, num_data=2*(10**4))
    print(dataset['truncrated'][9999])
    # Checking
    # plt.scatter(dataset['actions'][:, 0], dataset['actions'][:, 1])  # Show action distributions.
    # plt.plot(np.arange(len(dataset['rewards'])), dataset['rewards'])  # Show reward curve.
    plt.plot(np.arange(len(dataset['truncrated'])), dataset['truncrated'])
    plt.show()
    
