import numpy as np
from time import time, sleep

# ========================================= My Modules ===========================================
# from GPDP_v1.GPDP_v1 import GaussianProcessDiffusionPolicy
# from GPDQ.GPDQ import GaussianProcessDiffusionQlearning
# from SGDP.SGDP import StochasticGaussianprocessDiffusionPolicy
# from Diffusion_QL.Diffusion_QL import DiffusionQL
# # from SAC.sac import SAC
from environments.double_ring import DoubleRing
from dataset_creation import dataset_creation

# ======================================== Other Modules =========================================
from sys import platform
import os
import argparse
import random
import torch
CUDA = torch.cuda.is_available()
import matplotlib.pyplot as plt

# ======================================= Local Functions ========================================
def addArguments(parser):
    parser.add_argument('--env', default='double_ring', help='Environment: ["double_ring", "bottleneck", "intersection"], default:"double_ring"')
    parser.add_argument('--density', default=44, help='The density of the environment (number of vehicle), default:44')
    parser.add_argument('--alg', default="SGDP", help='Algorithm: ["SAC-S", "SAC-D", "D-QL", "GPDP"], default:GPDP')
    parser.add_argument('--task', default='training', help='Task: ["training", "testing", "dataset"], default:training')
    parser.add_argument('--gradient_step', default=100000, help='Gradient Step, default:1000000')
    parser.add_argument('--eval_mode', default='standard', help='Evaluation mode: [standard, shifted], default:standard')
    parser.add_argument('--rendering', default='no-render', help='Rendering: [render, no-render], default:no-render')

def setGlobalSeed(seed:int):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

def getDataset(path:str):
    if os.path.isfile(path):
        if args.task != 'dataset':
            # Load the dataset
            dataset = np.load(path)
        else:
            # Check if you want to override the previous dataset. 
            _override_ans = input('{:s} The dataset is already existed. Do you want to override it ? '.format(caution_tag))
            if _override_ans == 'y' or _override_ans =='Y' or _override_ans =='yes' or _override_ans =='YES' or _override_ans =='Yes':
                _dataset_creation_flag = True
            else:
                _dataset_creation_flag = False
                print(f'{main_tag}', 'The dataset is already existed and no override command. Exit gracefully.')
    else:
        if args.task != 'dataset':
            # Error, print that the user should create the dataset first. 
            print(f'{caution_tag}', 'The dataset is not found. Please create the dataset first by inputing \'--task dataset\'.')
            _dataset_creation_flag = False
            exit()
        else:
            _dataset_creation_flag = True
        
    if _dataset_creation_flag:
        # # Ask for the number of data points. 
        # print('{:s} Please input the number of data points...'.format(main_tag))
        # _num_data = int(input(''))
        _num_data = int(input('{:s} Please input the number of data points... '.format(main_tag)))
        # Create a new dataset.
        dataset = dataset_creation.datasetCreation(env=env, num_data=_num_data)

        # Saving the dataset.
        np.savez(path, dataset)
        
        # Print Final status
        print(f'{main_tag}', 'The dataset is successfully created and saved to ', f'{path}')

        # # Checking
        # # plt.scatter(dataset['actions'][:, 0], dataset['actions'][:, 1])  # Show action distributions.
        # # plt.plot(np.arange(len(dataset['rewards'])), dataset['rewards'])  # Show reward curve.
        # plt.plot(np.arange(len(dataset['truncrated'])), dataset['truncrated'])
        # plt.show()
    
# ====================================== Main Program Start ======================================
if __name__ == '__main__':
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
    main_tag = '(main) ----->'
    caution_tag = '(main) !!!!!>'

    # ========== Create objects
    env = DoubleRing(num_veh=NUM_VEH, render=RENDERING)  # Environment object

    # ========== Check whether the dataset/replay_buffers is exist ?
    # ========== The dataset or replay buffers are required to intialise the RL agent. 
    # !!!!!!!!!! For the dataset path, I will change to desktop location when I publish this online. 
    if platform == 'linux':
        dataset_path = '/home/amornyos/PhD/Packages/resources/datasets/{}_dataset.npz'.format(ENV_NAME)  # Linux
    elif platform == 'win32':
        dataset_path = 'C:/Amornyos/PhD/resources/datasets/{}_dataset.npz'.format(ENV_NAME)  # For Windows
    elif platform == 'darwin':
        dataset_path = '/Users/amornyos/MHOO_on_MAC/PhD/Thesis/Resources/datasets/{}_dataset.npz'.format(ENV_NAME)  # For macOS
    # Get the dataset
    dataset = getDataset(dataset_path)


