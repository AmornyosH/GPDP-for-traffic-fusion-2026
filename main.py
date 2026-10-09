import numpy as np
from time import time, sleep

# ========================================= My Modules ===========================================
from environments.double_ring import DoubleRing
from utilities import dataset_creation, evaluation, miscellaneous as misc, training
from utilities import plot_results, plot_actions

# ======================================== Other Modules =========================================
from sys import platform
import os
import argparse
import random
import torch
import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser(description='GPDP IEEE_Trans_ITS PyTorch Args')
    parser.add_argument('--env', default='double_ring', help='Environment: ["double_ring", "bottleneck", "intersection"], default:"double_ring"')
    parser.add_argument('--density', default=44, help='The density of the environment (number of vehicle), default:44')
    parser.add_argument('--alg', default="GPDP", help='Algorithm: ["GPDP", "SGDQ", "derived", "IDM"], default:GPDP')
    parser.add_argument('--task', default='testing', help='Task: ["training", "testing", "dataset"], default:training')
    parser.add_argument('--dataset_type', default='expert')
    parser.add_argument('--gradient_step', default=200000, help='Gradient Step, default:1000000')
    parser.add_argument('--eval_mode', default='standard', help='Evaluation mode: [standard, shifted], default:standard')
    parser.add_argument('--render', action='store_true', help='render the simulation')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--dataset_path', default=None, help='Override path for dataset directory (e.g. /data/datasets)')
    parser.add_argument('--train_path', default=None, help='Override path for training records directory')
    parser.add_argument('--eval_path', default=None, help='Override path for evaluation results directory')
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
    dataset_path, training_record_path, evaluation_path = misc.getPaths(
        operating_system=platform, alg=args.alg, env_name=ENV_NAME,
        dataset_path=args.dataset_path, train_path=args.train_path, eval_path=args.eval_path)

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

    # ========== Action Collection Tasks (Visualisation)
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
        speed_plots = plot_results.ResultsPlots()
        _veh_density = [(i * 1000 / 500) for i in env.num_veh_config]

        # Load IDM results
        _idm_results_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/IDM_double_ring_evaluation_results'
        _idm_results_path += '_' + args.dataset_type
        _idm_results_path += '_' + args.eval_mode
        idm_results = np.load(_idm_results_path+'.npz', allow_pickle=True)
        speed_plots.plot_speed_results(results=idm_results, density=_veh_density, label='IDM', color='red')

        # Load derived results
        _derived_results_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/derived_double_ring_evaluation_results'
        _derived_results_path += '_' + args.dataset_type
        _derived_results_path += '_' + args.eval_mode
        derived_results = np.load(_derived_results_path+'.npz', allow_pickle=True)
        speed_plots.plot_speed_results(results=derived_results, density=_veh_density, label='derived', color='orange')

        # Load TRPO results
        # _trpo_results_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/trpo_results_shifted'
        _trpo_results_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/trpo_results_standard'
        # _trpo_results_path += '_' + args.dataset_type
        # _trpo_results_path += '_' + args.eval_mode
        trpo_results = np.load(_trpo_results_path+'.npz', allow_pickle=True)
        speed_plots.plot_speed_results(results=trpo_results, density=_veh_density, label='TRPO', color='purple')
        
        # Load our proposed policy results
        _loaded_path = evaluation_path
        _loaded_path += '_' + args.dataset_type
        _loaded_path += '_' + args.eval_mode
        gpdp_results = np.load(_loaded_path+'.npz', allow_pickle=True)
        speed_plots.plot_speed_results(results=gpdp_results, density=_veh_density, label='Ours', color='blue')

        speed_plots.show_plots(save_path=f'figures/speed_results_{args.eval_mode}.png')

    # ========== Results Plotting (Visualisation)
    if args.task == 'plot_action':
        action_plots = plot_actions.ActionsPlots()
        _veh_density = [(i * 1000 / 500) for i in env.num_veh_config]

        # Load actions from the dataset (subplot-1)
        dataset_actions = dataset['actions'][0:]
        action_plots.plot_dataset_actions(results=dataset_actions, label='$\mathbf{a}\sim\mathcal{D}$')

        # Load actions from the TRPO policy (subplot-1)
        _trpo_results = np.load('C:/Amornyos/PhD/resources/T-ITS/training_records/trpo_action_collection.npy', allow_pickle=True)
        action_plots.plot_actions(results=_trpo_results, label='TRPO')

        # Load actions form our proposed policy (subplot-2)
        _loaded_path = evaluation_path
        _loaded_path += '_' + args.dataset_type
        _loaded_path += '_' + args.eval_mode
        _loaded_path += '_actions_collection'
        gpdp_results = np.load(_loaded_path+'.npy', allow_pickle=True)
        action_plots.plot_actions(results=gpdp_results, label='Ours')
        action_plots.show_plots(save_path=f'figures/action_distribution_{args.eval_mode}.png')
        
