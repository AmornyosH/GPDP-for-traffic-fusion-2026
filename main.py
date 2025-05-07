import numpy as np
from time import time, sleep

# ========================================= My Modules ===========================================
from GPDP_v1.GPDP_v1 import GaussianProcessDiffusionPolicy
from GPDQ.GPDQ import GaussianProcessDiffusionQlearning
from SGDP.SGDP import StochasticGaussianprocessDiffusionPolicy
from Diffusion_QL.Diffusion_QL import DiffusionQL
# from SAC.sac import SAC

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

def createDataset():
    _x = []
    for _ in range(100):
        _x.append(np.arange(start=0, stop=4*np.pi, step=0.05))
    _x = np.hstack(_x)
    _y = np.zeros([len(_x)])
    _r = np.zeros([len(_y)])  

    for i in range(len(_y)):
        if i <= len(_y)//2:
            _y[i] = np.sin(_x[i]) + np.random.normal(0, 0.05, size=1)
        else:
            _y[i] = -np.sin(_x[i]) + np.random.normal(0, 0.05, size=1)

    for j in range(len(_r)):
        if _x[j] <= 2*np.pi:
            _r[j] = _y[j] * 2
        else: 
            _r[j] = -_y[j]
        # print('transition: ', j, _x[j], _y[j], _r[j])

    _x_p_1 = np.zeros([len(_x)])
    _x_p_1[0:-2] = _x[1:-1]
    _x_p_1[-1] = _x[0]

    np.savez(EXPERT_PATH, 
             len(_x),      # arr_0
             _x.reshape(-1, 1),           # arr_1
             _y.reshape(-1, 1),           # arr_2
             _r.reshape(-1, 1),           # arr_3
             _x_p_1.reshape(-1, 1))       # arr_4

def getParamsDict(env):
    return {'environment': env,
            'epoch': 256,
            'num_batches': 10,
            'expert_num_batches': 10,
            'horizon': 1000,
            'action_config': 'continous',
            'state_dim': 1,
            'action_dim': 1, 
            'task': TASK}

def evaluatingSine(eval_times:int=1, shift:bool=False):
    _step = 0
    _s_t = 0
    _x_test = []
    _y_test = []

    while _step <= 2*np.pi:
        _pred = agent.predict(state=_s_t, size=1)
        # _pred = agent.predict(state=_s_t, size=1, guide=False)
        _x_test.append(_s_t)
        _y_test.append(_pred.tolist())
        _step += 0.01
        if shift:
            _s_t = _step if _step <= np.pi else _step + (2*np.pi)
        else:
            _s_t = _step
    
    return np.vstack(_x_test), np.vstack(_y_test)
        
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
RENDER = True if args.rendering == 'render' else False
UNCERTAINTY = True if args.eval_mode == 'shifted' else False

# Get parameters dictionary of the environment.
params_dict = getParamsDict(env=ENV_NAME)
if not os.path.isfile(EXPERT_PATH):
    createDataset()
dataset = np.load(EXPERT_PATH)  # ========== Load the dataset

# ========== Create the agent
if args.alg == 'GPDP':
    agent = GaussianProcessDiffusionPolicy(params_dict=params_dict, dataset=dataset, ft=False)
elif args.alg == 'SGDP':
    agent = StochasticGaussianprocessDiffusionPolicy(params_dict=params_dict, dataset=dataset, ft=False)
elif args.alg == 'GPDQ':
    agent = GaussianProcessDiffusionQlearning(params_dict=params_dict, dataset=dataset, ft=False)
elif args.alg == 'D-QL':
    agent = DiffusionQL(params_dict=params_dict, dataset=dataset, ft=False)
# elif args.alg == 'SAC-D':
#     agent = SAC(load=True, params_dict=params_dict, eval_policy=True, cuda=CUDA)
# elif args.alg == 'SAC-S':
#     agent = SAC(load=True, params_dict=params_dict, eval_policy=False, cuda=CUDA)

# ========== Training (Offline)
EPOCH = int(args.gradient_step) / (dataset['arr_0']//agent.MINIBATCH_SIZE)
if args.task == 'training':
    # Train Diffusion Policy and Value functions.
    while agent.training_record < EPOCH:
        print('Start Offline Training...')
        # Train Diffusion Policy
        agent.training(total_epoch=EPOCH, eval=False)

    # # Check if the agent has gp ?
    # if hasattr(agent, 'gp_model'):
    #     # Train GP (after)
    #     agent.gp_model.y_train = agent.getAlteredObservation(agent.gp_model.x_train)
    #     # agent.gp_model.q_mean = torch.nn.Parameter(agent.gp_model.y_train[agent.gp_model.indexes], requires_grad=True)
    #     agent.gp_model.myTraining(total_epoch=2000, ft=False)
    print('Training is done!')

# ========== Testing
if args.task == 'testing':
    # ---------- Simple sine waves task ----------
    if args.env == 'simple_sine':
        x_test, y_test = evaluatingSine(eval_times=1, shift=False)
        # x_test, y_test = evaluatingSine(eval_times=1, shift=True)
        np.savez(agent.evaluation_path, x_test, y_test)

        result = np.load(agent.evaluation_path)
        plt.scatter(dataset['arr_1'], dataset['arr_2'], label='gt')
        # plt.scatter(dataset['arr_1'], dataset['arr_3'], alpha=0.1, label='gt_reward')
        plt.scatter(result['arr_0'], result['arr_1'], alpha=1.0, label='ours')
        plt.legend()
        plt.show()
