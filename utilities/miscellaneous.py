
import json
import numpy as np
import torch
import os

MISC_TAG = '(MISC) ----->'
MISC_CAUTION_TAG = '(MISC) !!!!!>'

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')

def _loadConfig() -> dict:
    """Read config.json from the project root. Returns empty dict if missing."""
    if os.path.isfile(_CONFIG_FILE):
        with open(_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

# Paths Management Method
def getPaths(operating_system:str, alg:str, env_name:str,
             dataset_path:str=None, train_path:str=None, eval_path:str=None):
    """
    Resolve the three resource paths used throughout the project.

    Priority (highest → lowest):
      1. CLI flags  (--dataset_path / --train_path / --eval_path)
      2. config.json at the project root
      3. Per-machine hardcoded defaults
    """
    cfg = _loadConfig()

    # Merge: CLI flag beats config.json, config.json beats hardcoded default.
    _dp_root = dataset_path or cfg.get('dataset_path') or None
    _tp_root = train_path   or cfg.get('train_path')   or None
    _ep_root = eval_path    or cfg.get('eval_path')    or None

    # If all three roots are resolved (config or CLI), skip hardcoded defaults.
    if _dp_root and _tp_root and _ep_root:
        return (
            os.path.join(_dp_root, '{}_dataset'.format(env_name)),
            os.path.join(_tp_root, '{}_{}_training_records'.format(alg, env_name)),
            os.path.join(_ep_root, '{}_{}_evaluation_results'.format(alg, env_name)),
        )

    # Hardcoded per-machine defaults.
    if operating_system == 'linux':
        _base = '/home/amornyos/PhD/Packages/resources/datasets'
        _dp   = os.path.join(_base, '{}_dataset'.format(env_name))
        _tp   = os.path.join(_base, '{}_{}_training_records'.format(alg, env_name))
        _ep   = os.path.join(_base, '{}_{}_evaluation_results'.format(alg, env_name))
    elif operating_system == 'win32':
        if os.path.isdir('C:/Amornyos/PhD/resources/T-ITS'):
            _ds = 'C:/Amornyos/PhD/resources/T-ITS/datasets'
            _tr = 'C:/Amornyos/PhD/resources/T-ITS/training_records'
        elif os.path.isdir('C:/Users/AmornyosH/resources/T-ITS'):
            _ds = 'C:/Users/AmornyosH/resources/T-ITS/datasets'
            _tr = 'C:/Users/AmornyosH/resources/T-ITS/training_records'
        else:
            raise RuntimeError(
                'No default path found for this machine. '
                'Fill in config.json at the project root, or pass '
                '--dataset_path, --train_path and --eval_path.')
        _dp = os.path.join(_ds, '{}_dataset'.format(env_name))
        _tp = os.path.join(_tr, '{}_{}_training_records'.format(alg, env_name))
        _ep = os.path.join(_tr, '{}_{}_evaluation_results'.format(alg, env_name))
    elif operating_system == 'darwin':
        _base = '/Users/amornyos/MHOO_on_MAC/PhD/Thesis/Resources/datasets'
        _dp   = os.path.join(_base, '{}_dataset'.format(env_name))
        _tp   = os.path.join(_base, '{}_{}_training_records'.format(alg, env_name))
        _ep   = os.path.join(_base, '{}_{}_evaluation_results'.format(alg, env_name))
    else:
        raise RuntimeError(
            f'Unsupported platform "{operating_system}". '
            'Fill in config.json at the project root, or pass '
            '--dataset_path, --train_path and --eval_path.')

    # Partial overrides from config.json or CLI still apply.
    if _dp_root:
        _dp = os.path.join(_dp_root, '{}_dataset'.format(env_name))
    if _tp_root:
        _tp = os.path.join(_tp_root, '{}_{}_training_records'.format(alg, env_name))
    if _ep_root:
        _ep = os.path.join(_ep_root, '{}_{}_evaluation_results'.format(alg, env_name))

    return _dp, _tp, _ep

# Range Conversion Method
def rangeConversion(input, new_min, new_max, old_min=-1, old_max=1) -> float:
    '''
    Range Conversion Method (Mapping from old range to a new range) \\
    This method will map the value in the defined old range (default = [-1, 1])
    in to a new defined range (no default). 
    Formula = new_min + ((input - old_min) * (new_max - new_min) / (old_max - old_min))
    
    Input Arguments: sample:Any, new_min, new_max, old_min=-1, old_max=1
    Return: sample in new range (sample \in [new_min, new_max])
    '''
    # Default of old range is [-1, 1]
    # Define parameters
    new_min = torch.tensor(new_min, dtype=torch.float32)
    new_max = torch.tensor(new_max, dtype=torch.float32)
    old_min = torch.tensor(old_min, dtype=torch.float32)
    old_max = torch.tensor(old_max, dtype=torch.float32)
    
    return torch.clip(new_min + ((input - old_min) * (new_max - new_min) / (old_max - old_min)), 
                            min=new_min, 
                            max=new_max)

# Best Trajectory Extraction Method
def bestTrajExtraction(params_dict:dict, dataset:dict):
    _num_samples = dataset['num_sample']
    _state = dataset['observations']
    _action = dataset['actions']
    _reward = dataset['rewards']
    _next_state = dataset['next_observations']
    _truncrated = dataset['truncrated']
    _terminated = dataset['terminations']
    _accum_reward = 0
    _best_reward = 0
    _counter = 0
    for n in range(_num_samples):
        if _truncrated[n] or _terminated[n]:
            if _accum_reward > _best_reward:
                _best_reward = _accum_reward
                _last_idx = n
                _first_idx = n - _counter
            # Reset
            _accum_reward = 0
            _counter = 0
        else:
            _accum_reward += _reward[n]
            _counter += 1
            if _counter >= params_dict['horizon']:
                # Reset
                _counter = 0
                _accum_reward = 0

    
    print(f'{MISC_TAG} Best_reward: ', _best_reward, ', First_idx: ', _first_idx, ', Last_idx: ', _last_idx)

    _best_dataset = ({'num_sample': len(_state[_first_idx:_last_idx]), 
                      'observations': _state[_first_idx:_last_idx+1], 
                      'actions': _action[_first_idx:_last_idx+1], 
                      'rewards': _reward[_first_idx:_last_idx+1], 
                      'next_observations': _next_state[_first_idx:_last_idx+1], 
                      'truncrated': _truncrated[_first_idx:_last_idx+1],
                      'termination': _terminated[_first_idx:_last_idx+1]})

    return _best_dataset

# Best Trajectory Extraction Method
def allTrajExtraction(params_dict:dict, dataset:dict):
    # Reduce the resolution of the dataset
    _num_skip = 1
    _num_samples = 50000
    _start_point = 0
    # _new_indices = np.arange(dataset['num_sample'], step=_num_skip)
    _new_indices = np.arange(_num_samples, step=_num_skip)
    _new_horizon = params_dict['horizon']//_num_skip
    _state = dataset['observations'][_start_point:_start_point+_num_samples]
    _action = dataset['actions'][_start_point:_start_point+_num_samples]
    _reward = dataset['rewards'][_start_point:_start_point+_num_samples]
    _next_state = dataset['next_observations'][_start_point:_start_point+_num_samples]
    _truncrated = dataset['truncrated'][_start_point:_start_point+_num_samples]
    _terminated = dataset['terminations'][_start_point:_start_point+_num_samples]
    # _num_samples = len(_state)
    _accum_reward = 0
    _best_reward = 0
    _counter = 0

    # for n in range(_num_samples):
    #     if _truncrated[n] or _terminated[n] or _counter >= _new_horizon:
    #         # Reset
    #         _accum_reward = 0
    #         _counter = 0
    #     else:
    #         _accum_reward += _reward[n]
    #         _counter += 1

    # print(f'{MISC_TAG} Best_reward: ', _best_reward, ', First_idx: ', _first_idx, ', Last_idx: ', _last_idx)

    _best_dataset = ({'num_sample': len(_state), 
                      'observations': _state, 
                      'actions': _action, 
                      'rewards': _reward, 
                      'next_observations': _next_state, 
                      'truncrated': _truncrated,
                      'termination': _terminated})
    # print(np.sum(_reward))
    return _best_dataset

# Interval Trajectory Extraction For GP Method
def IntTrajExtraction(interval:int, params_dict:dict, dataset:dict):
    _num_samples = dataset['num_sample']
    _state = dataset['observations']
    _action = dataset['actions']
    _reward = dataset['rewards']
    _next_state = dataset['next_observations']
    _truncrated = dataset['truncrated']
    _terminated = dataset['terminations']

    _states_append = []
    _actions_append = []
    _rewards_append = []
    _next_states_append = []

    _accum_reward = 0
    _best_reward = 0
    _counter = 0
    _stop = False
    for n in range(_num_samples):
        if _truncrated[n] or _terminated[n]:
            if _accum_reward > _best_reward:
                _best_reward = _accum_reward
                _last_idx = n
                _first_idx = n - _counter
            # Reset
            _accum_reward = 0
            _counter = 0
        else:
            # _accum_reward += _reward[n]
            _states_append.append(_state[n])
            _actions_append.append(_action[n])
            _rewards_append.append(_reward[n])
            _next_states_append.append(_next_state[n])
            _counter += 1
            if _counter >= params_dict['horizon']:
                # Reset
                _counter = 0
                _accum_reward = 0

    print('best_reward: ', _best_reward, ', first_idx: ', _first_idx, ', last_idx: ', _last_idx)

    _best_dataset = ({'num_sample': len(_state[_first_idx:_last_idx]), 
                      'observations': _state[_first_idx:_last_idx], 
                      'actions': _action[_first_idx:_last_idx], 
                      'rewards': _reward[_first_idx:_last_idx], 
                      'next_observations': _next_state[_first_idx:_last_idx], 
                      'truncrated': _truncrated[_first_idx:_last_idx],
                      'termination': _terminated[_first_idx:_last_idx]})

    return _best_dataset

# 
def klDivergence(input_1, input_2):
    '''
    KL-Divergence Computation Method \\
    This method will return the Kullback-Leibler Divergence (D_KL) between input_1 and input_2.\\
    Remind that D_KL between input_1 and input_2 is not equal to D_KL between input_2 and input_1.

    Input Arguments: input_1:Any, input_2:Any (Numpy Format)
    Return: D_KL(input_1 || input_2) (Numpy Format)
    '''
    # old_policy = tf.gather_nd(old_policy, indices=slice_indices.astype(int))
    # return tf.reduce_mean(-tf.reduce_sum(old_policy * tf.math.log(new_policy/old_policy), axis=1)).numpy()
    kl = -np.sum(input_1 * np.log(input_2/input_1)) / len(input_1)
    return kl

# Sinusoidal Position Encoding (Same as the original Transformer.)
def sinPositionEncoding(seq_len, dim, N=10000):
    output = np.zeros([seq_len, dim], dtype=float)
    for k in range(seq_len):
        for i in np.arange(int(dim/2)):
            denominator = np.power(N, 2*i/dim)
            output[k, 2*i] = np.sin(k/denominator)
            output[k, 2*i+1] = np.cos(k/denominator)
    return output

# Positional encoding method (My implementation)
def sinusoidalEncoding(input, seq_len):
    t_space = np.linspace(start=-1, stop=1, num=seq_len, dtype=float)  # [-1, -0.5, 0, 0.5, 1]
    return np.sin(2 * np.pi * t_space[input-1])  # sin(2 * pi * t)

