import numpy as np
from time import time, sleep

# ========================================= My Modules ===========================================
# from GPDP_v1.GPDP_v1 import GaussianProcessDiffusionPolicy
# from GPDQ.GPDQ import GaussianProcessDiffusionQlearning
# from SGDP.SGDP import StochasticGaussianprocessDiffusionPolicy
# from Diffusion_QL.Diffusion_QL import DiffusionQL
# # from SAC.sac import SAC
from environments.double_ring import DoubleRing

# ======================================== Other Modules =========================================
import os
import argparse
import random
import torch
CUDA = torch.cuda.is_available()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# ======================================= Local Functions ========================================
def addArguments(parser):
    parser.add_argument('--env', default='simple_sine', help='Environment: ["simple_sine", "Walker2d-v5"], default:"simple_sine"')
    parser.add_argument('--alg', default="GPDQ", help='Algorithm: ["SAC-S", "SAC-D", "D-QL", "GPDP"], default:GPDP')
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

parser = argparse.ArgumentParser(description='SGDP (Neurips2025) PyTorch Args')
addArguments(parser)
args = parser.parse_args()

# ========== Global constants
ENV_NAME = args.env
TASK = args.task
EXPERT_PATH = 'datasets/dataset_{}.npz'.format(ENV_NAME)
RENDERING = True if args.rendering == 'render' else False
UNCERTAINTY = True if args.eval_mode == 'shifted' else False

av_id = 'AV_1'

# ========== Test SUMO
# Initialise environment
env = DoubleRing(num_veh=50, render=RENDERING)
state = env.reset(veh_id=av_id)
env.close()
