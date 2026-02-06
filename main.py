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
# import seaborn as sns
# sns.set_theme()

# ======================================= Local Functions ========================================
def getDataset(path:str, type:str):
    _path = path+'_'+type
    if os.path.isfile(_path+'.npz'):
        if args.task != 'dataset':
            # Load the dataset
            dataset = np.load(_path+'.npz', allow_pickle=True)
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
        
        print(f'{main_tag} The dataset ({type}) is being created.')

        # Create a new dataset.
        dataset = dataset_creation.datasetCreation(env=env, num_data=_num_data, type=type)

        # Saving the dataset.
        np.savez(_path, **dataset)
        
        # Print Final status
        print(f'{main_tag}', 'The dataset is successfully created and saved to ', f'{_path}')

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
    parser.add_argument('--dataset_type', default='expert')
    parser.add_argument('--gradient_step', default=200000, help='Gradient Step, default:1000000')
    parser.add_argument('--eval_mode', default='standard', help='Evaluation mode: [standard, shifted], default:standard')
    parser.add_argument('--render', action='store_true', help='render the simulation')
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    # ========== Define global constants
    ENV_NAME = args.env
    NUM_VEH = int(args.density)
    TASK = args.task
    EXPERT_PATH = 'datasets/dataset_{}'.format(ENV_NAME)
    UNCERTAINTY = True if args.eval_mode == 'shifted' else False
    RENDERING = args.render
    av_id = 'AV_1'
    main_tag = '(main) ----->'
    caution_tag = '(main) !!!!!>'
    
    # ========== Create environment object
    env = DoubleRing(render=RENDERING, cond=args.eval_mode)  # Environment object
    
    # ---------- Define paths
    dataset_path, training_record_path, evaluation_path = misc.getPaths(operating_system=platform, alg=args.alg, env_name=ENV_NAME)

    # ========== Check whether the dataset/replay_buffers is exist ?
    # !!!!!!!!!! The dataset or replay buffers are required to intialise the RL agent. 
    # Get the dataset
    dataset = getDataset(dataset_path, args.dataset_type)

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
                                                       num_veh=int(args.density),
                                                       alg=args.alg,
                                                       cont=False, 
                                                       dataset=dataset, 
                                                       total_steps=int(args.gradient_step), 
                                                       imposter=UNCERTAINTY)
    
    # ========== Testing (Evaluation)
    if args.task == 'testing':
        _total_return = []
        _total_avg_av_speed = []
        _total_avg_all_speed = []
        _total_std_av_speed = []
        _total_std_all_speed = []

        for item in env.num_veh_config:
            _eval_results = evaluation.sumoInteraction(env=env,
                                                       num_veh=item,
                                                       alg=args.alg,
                                                       dataset=dataset,
                                                       load=args.load, 
                                                       imposter=UNCERTAINTY, 
                                                       eval_time=1 if args.alg == 'derived' or args.alg == 'IDM' else 10)
            _total_return.append(_eval_results['mean_accum_reward'])
            _total_avg_av_speed.append(_eval_results['mean_av_speed'])
            _total_avg_all_speed.append(_eval_results['mean_all_speed'])
            _total_std_av_speed.append(_eval_results['std_av_speed'])
            _total_std_all_speed.append(_eval_results['std_all_speed'])
        
            print(f'{main_tag}', 'Evaluation for', f'{args.alg}', ' with ', f'{item}', ' is completed.')
            print(f'{main_tag}', 'The return is ', _eval_results['mean_accum_reward'], '.')
            print(f'{main_tag}', 'The average AV speed is ', _eval_results['mean_av_speed'], ', std: ', _eval_results['std_av_speed'], '.')
            print(f'{main_tag}', 'The average All Vehicle speed is ', _eval_results['mean_all_speed'], ', std: ', _eval_results['std_all_speed'], '.')
            print(f'{main_tag}', 'Exit Gracefully. Goodbye.')

        # Save the evaluation records
        _saved_path = evaluation_path
        _saved_path += '_' + args.dataset_type
        # _saved_path += ''_shifted' if UNCERTAINTY else '_standard''
        _saved_path += '_' + args.eval_mode
        np.savez(_saved_path, 
                {'num_veh': len(env.veh_id_list), 
                'accum_reward': np.vstack(_total_return),
                'avg_av_speed': np.vstack(_total_avg_av_speed), 
                'avg_all_speed': np.vstack(_total_avg_all_speed), 
                'std_av_speed': np.vstack(_total_std_av_speed), 
                'std_all_speed': np.vstack(_total_std_all_speed)})

    if args.task == 'actions_collection':
        _total_return = []
        _total_avg_av_speed = []
        _total_avg_all_speed = []
        _total_std_av_speed = []
        _total_std_all_speed = []

        _eval_results = evaluation.sumoInteraction(env=env,
                                                    num_veh=44,
                                                    alg=args.alg,
                                                    dataset=dataset,
                                                    load=args.load, 
                                                    imposter=UNCERTAINTY, 
                                                    eval_time=1)
        # _total_return.append(_eval_results['mean_accum_reward'])
        # _total_avg_av_speed.append(_eval_results['mean_av_speed'])
        # _total_avg_all_speed.append(_eval_results['mean_all_speed'])
        # _total_std_av_speed.append(_eval_results['std_av_speed'])
        # _total_std_all_speed.append(_eval_results['std_all_speed'])
        
        print(f'{main_tag}', 'Evaluation for', f'{args.alg}', ' with ', 44, ' is completed.')
        print(f'{main_tag}', 'The return is ', _eval_results['mean_accum_reward'], '.')
        print(f'{main_tag}', 'The average AV speed is ', _eval_results['mean_av_speed'], ', std: ', _eval_results['std_av_speed'], '.')
        print(f'{main_tag}', 'The average All Vehicle speed is ', _eval_results['mean_all_speed'], ', std: ', _eval_results['std_all_speed'], '.')
        print(f'{main_tag}', 'Exit Gracefully. Goodbye.')

        # Save the evaluation records
        _saved_path = evaluation_path
        _saved_path += '_' + args.dataset_type
        # _saved_path += ''_shifted' if UNCERTAINTY else '_standard''
        _saved_path += '_' + args.eval_mode
        _saved_path += '_' + args.task

        np.save(_saved_path, _eval_results['action'])

    # ========== Results Plotting (Visualisation)
    if args.task == 'plot':

        _veh_density = [(i * 1000 / 500) for i in env.num_veh_config]

        _idm_results_path = '/home/amornyos/PhD/Packages/resources/datasets/IDM_double_ring_evaluation_results'
        _idm_results_path += '_' + args.dataset_type
        # _idm_results_path += '_shifted' if UNCERTAINTY else '_standard'
        _idm_results_path += '_' + args.eval_mode
        idm_results = np.load(_idm_results_path+'.npz', allow_pickle=True)
        idm_results = idm_results['arr_0'].item()
        idm_av_speed_mean = idm_results['avg_av_speed'] * 3600 / 1000
        idm_av_speed_std = idm_results['std_av_speed'] * 3600 / 1000
        idm_all_speed_mean = idm_results['avg_all_speed'] * 3600 / 1000
        idm_all_speed_std = idm_results['std_all_speed'] * 3600 / 1000

        _derived_results_path = '/home/amornyos/PhD/Packages/resources/datasets/derived_double_ring_evaluation_results'
        _derived_results_path += '_' + args.dataset_type
        # _derived_results_path += '_shifted' if UNCERTAINTY else '_standard'
        _derived_results_path += '_' + args.eval_mode
        derived_results = np.load(_derived_results_path+'.npz', allow_pickle=True)
        derived_results = derived_results['arr_0'].item()
        derived_av_speed_mean = derived_results['avg_av_speed'] * 3600 / 1000
        derived_av_speed_std = derived_results['std_av_speed'] * 3600 / 1000
        derived_all_speed_mean = derived_results['avg_all_speed'] * 3600 / 1000
        derived_all_speed_std = derived_results['std_all_speed'] * 3600 / 1000

        _loaded_path = evaluation_path
        _loaded_path += '_' + args.dataset_type
        # _loaded_path += '_shifted' if UNCERTAINTY else '_standard'
        _loaded_path += '_' + args.eval_mode
        gpdp_results = np.load(_loaded_path+'.npz', allow_pickle=True)
        gpdp_results = gpdp_results['arr_0'].item()
        gpdp_av_speed_mean = np.squeeze(gpdp_results['avg_av_speed']) * 3600 / 1000
        gpdp_av_speed_std = np.squeeze(gpdp_results['std_av_speed']) * 3600 / 1000
        gpdp_all_speed_mean = np.squeeze(gpdp_results['avg_all_speed']) * 3600 / 1000
        gpdp_all_speed_std = np.squeeze(gpdp_results['std_all_speed']) * 3600 / 1000


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
        # ========= AX1: Plot all vehicles speed (Left Figure)
        # IDM
        ax1.scatter(_veh_density, idm_all_speed_mean, color='red')
        ax1.plot(_veh_density, idm_all_speed_mean, color='red', label='IDM')
        ax1.fill_between(_veh_density, 
                         np.squeeze(idm_all_speed_mean+idm_all_speed_std), 
                         np.squeeze(idm_all_speed_mean-idm_all_speed_std), color='red', alpha=0.10)
        # Derived
        ax1.scatter(_veh_density, derived_all_speed_mean, color='orange')
        ax1.plot(_veh_density, derived_all_speed_mean, color='orange', label='Derived')
        ax1.fill_between(_veh_density, 
                         np.squeeze(derived_all_speed_mean+derived_all_speed_std), 
                         np.squeeze(derived_all_speed_mean-derived_all_speed_std), color='orange', alpha=0.10)
        # GPDP
        ax1.scatter(_veh_density, gpdp_all_speed_mean, color='blue')
        ax1.plot(_veh_density, gpdp_all_speed_mean, color='blue', label='Ours')
        ax1.fill_between(_veh_density, 
                         gpdp_all_speed_mean+gpdp_all_speed_std, 
                         gpdp_all_speed_mean-gpdp_all_speed_std, color='blue', alpha=0.10)

        ax1.set_ylim(0, 4*3600/1000)
        ax1.grid(True)
        ax1.legend(fontsize=16, loc='upper right')
        ax1.set_title('Average All Vehicle Speed')

        # ========= AX2: Plot AV speed (Right Figure)
        # IDM
        ax2.scatter(_veh_density, idm_av_speed_mean, color='red')
        ax2.plot(_veh_density, idm_av_speed_mean, color='red', label='IDM')
        ax2.fill_between(_veh_density, 
                         np.squeeze(idm_av_speed_mean+idm_av_speed_std), 
                         np.squeeze(idm_av_speed_mean-idm_av_speed_std), color='red', alpha=0.10)
        # Derived
        ax2.scatter(_veh_density, derived_av_speed_mean, color='orange')
        ax2.plot(_veh_density, derived_av_speed_mean, color='orange', label='Derived')
        ax2.fill_between(_veh_density, 
                         np.squeeze(derived_av_speed_mean+derived_av_speed_std), 
                         np.squeeze(derived_av_speed_mean-derived_av_speed_std), color='orange', alpha=0.10)
        # GPDP
        ax2.scatter(_veh_density, gpdp_av_speed_mean, color='blue')
        ax2.plot(_veh_density, gpdp_av_speed_mean, color='blue', label='Ours')
        ax2.fill_between(_veh_density, 
                         gpdp_av_speed_mean+gpdp_av_speed_std, 
                         gpdp_av_speed_mean-gpdp_av_speed_std, color='blue', alpha=0.10)

        ax2.set_ylim(0, 4*3600/1000)
        ax2.grid(True)
        ax2.legend(fontsize=16, loc='upper right')
        ax2.set_title('Average AV Speed')

        # plt.ylim(0, 6* 3600 / 1000)
        # plt.xlabel('Vehicle Density (veh/km)')
        # plt.ylabel('All Vehicle Speed (km/hr)')
        # plt.title('Average AV Speed')
        # plt.legend()
        # plt.grid(True)
        fig.supxlabel('Vehicle Density (veh/km)', fontsize=20)
        fig.supylabel('Speed (km/hr)', fontsize=20)
        ax1.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='both', labelsize=14)
        # ax1.yticks(fontsize=14)
        # ax2.xticks(fontsize=14)
        # ax2.yticks(fontsize=14)
        plt.tight_layout()
        plt.show()

    # ========== Results Plotting (Visualisation)
    if args.task == 'plot_action':

        _veh_density = [(i * 1000 / 500) for i in env.num_veh_config]

        _loaded_path = evaluation_path
        _loaded_path += '_' + args.dataset_type
        _loaded_path += '_' + args.eval_mode
        _loaded_path += '_actions_collection'

        gpdp_results = np.load(_loaded_path+'.npy', allow_pickle=True)
        # gpdp_results = gpdp_results['arr_0'].item()
        action_0 = gpdp_results[:, 0]
        action_1 = gpdp_results[:, 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
        # ========= AX1: Plot all vehicles speed (Left Figure)
        # GPDP
        ax1.scatter(action_0, action_1, color='blue')

        # ax1.set_ylim(0, 6*3600/1000)
        # ax1.grid(True)
        # ax1.legend(fontsize=16, loc='upper right')

        # ========= AX2: Plot AV speed (Right Figure)

        # GPDP
        ax2.scatter(action_0, action_1, color='blue', alpha=0.2)

        # ax2.set_ylim(0, 6*3600/1000)
        # ax2.grid(True)
        # ax2.legend(fontsize=16, loc='upper right')

        # plt.ylim(0, 6* 3600 / 1000)
        # plt.xlabel('Vehicle Density (veh/km)')
        # plt.ylabel('All Vehicle Speed (km/hr)')
        # plt.title('Average AV Speed')
        # plt.legend()
        # plt.grid(True)
        fig.supxlabel('Acceleration Action $a_0$', fontsize=20)
        fig.supylabel('Lane Change Action $a_1$', fontsize=20)
        ax1.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='both', labelsize=14)

        # ax1.yticks(fontsize=14)
        # ax2.xticks(fontsize=14)
        # ax2.yticks(fontsize=14)

        ax1.grid(True)
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
