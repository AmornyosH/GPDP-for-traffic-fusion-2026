import numpy as np
from time import time, sleep

# ========================================= My Modules ===========================================
from environments.double_ring import DoubleRing
from utilities import dataset_creation, evaluation, miscellaneous as misc, training

# ======================================== Other Modules =========================================
from sys import platform
import os
import argparse
import random
import torch
# CUDA = torch.cuda.is_available()
CUDA = False
import matplotlib.pyplot as plt

# ======================================= Local Functions ========================================
def getDataset(path:str):
    if os.path.isfile(path):
        if args.task != 'dataset':
            # Load the dataset
            dataset = np.load(path, allow_pickle=True)
            _dataset_creation_flag = False
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
        np.savez(path, **dataset)
        
        # Print Final status
        print(f'{main_tag}', 'The dataset is successfully created and saved to ', f'{path}')

        # # Checking
        # # plt.scatter(dataset['actions'][:, 0], dataset['actions'][:, 1])  # Show action distributions.
        # # plt.plot(np.arange(len(dataset['rewards'])), dataset['rewards'])  # Show reward curve.
        # plt.plot(np.arange(len(dataset['truncrated'])), dataset['truncrated'])
        # plt.show()

    return dataset

# ====================================== Main Program Start ======================================
if __name__ == '__main__':
    # ========== Set global seed generator first. 
    seed = 2203
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    # ========== Define arguments parser. 
    parser = argparse.ArgumentParser(description='SGDP IEEE_Trans_ITS PyTorch Args')
    parser.add_argument('--env', default='double_ring', help='Environment: ["double_ring", "bottleneck", "intersection"], default:"double_ring"')
    parser.add_argument('--density', default=44, help='The density of the environment (number of vehicle), default:44')
    parser.add_argument('--alg', default="SGDP", help='Algorithm: ["SAC-S", "SAC-D", "D-QL", "GPDP"], default:GPDP')
    parser.add_argument('--task', default='testing', help='Task: ["training", "testing", "dataset"], default:training')
    parser.add_argument('--gradient_step', default=200000, help='Gradient Step, default:1000000')
    parser.add_argument('--eval_mode', default='standard', help='Evaluation mode: [standard, shifted], default:standard')
    parser.add_argument('--rendering', default='no-render', help='Rendering: [render, no-render], default:no-render')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    # ========== Define global constants
    ENV_NAME = args.env
    NUM_VEH = int(args.density)
    TASK = args.task
    EXPERT_PATH = 'datasets/dataset_{}'.format(ENV_NAME)
    UNCERTAINTY = True if args.eval_mode == 'shifted' else False
    RENDERING = True if args.rendering == 'render' else False
    av_id = 'AV_1'
    main_tag = '(main) ----->'
    caution_tag = '(main) !!!!!>'
    
    # ========== Create environment object
    env = DoubleRing(num_veh=NUM_VEH, render=RENDERING)  # Environment object
    
    # ---------- Define paths
    dataset_path, training_record_path = misc.getPaths(operating_system=platform, alg=args.alg, env_name=ENV_NAME)

    # ========== Check whether the dataset/replay_buffers is exist ?
    # !!!!!!!!!! The dataset or replay buffers are required to intialise the RL agent. 
    # Get the dataset
    dataset = getDataset(dataset_path)

    # ========== Training (Offline)
    if args.task == 'training':
        # Ask whether which training option is preferred ?
        _training_option = input('{:s} Please define training options ["offline", "online"]: '.format(main_tag))
        if _training_option == 'offline':
            # Go for offline training. 
            _training_record = training.offlineTraining(path=training_record_path, 
                                                                env=env,
                                                                alg=args.alg,
                                                                cont=args.load, 
                                                                dataset=dataset, 
                                                                total_steps=int(args.gradient_step))
            print(f'{main_tag}', 'Offline Training for', f'{args.alg}', 'is completed.')
            print(f'{main_tag}', 'The training record is save at the location: ', f'{training_record_path}')
            print(f'{main_tag}', 'Exit Gracefully. Goodbye.')
        if _training_option == 'online':
            _training_record = training.onlineTraining(path=training_record_path, 
                                                       env=env,
                                                       alg=args.alg,
                                                       cont=False, 
                                                       dataset=dataset, 
                                                       total_steps=int(args.gradient_step), 
                                                       imposter=UNCERTAINTY)
    
    # ========== Testing (Evaluation)
    if args.task == 'testing':
        _return = evaluation.sumoInteraction(env=env,
                                             num_veh=40,
                                             alg=args.alg,
                                             dataset=dataset,
                                             load=args.load, 
                                             imposter=UNCERTAINTY)
        print(f'{main_tag}', 'Evaluation for', f'{args.alg}', 'is completed.')
        print(f'{main_tag}', 'The return is ', f'{_return}', '.')
        print(f'{main_tag}', 'Exit Gracefully. Goodbye.')