import os
import torch
import numpy as np
from time import time

CUDA = torch.cuda.is_available()
if CUDA:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
# CUDA = False

class myExactGP(torch.nn.Module):
    def __init__(self, parent_alg:str, params_dict:dict, dataset:dict, load:bool=False, cuda:bool=False, online:bool=False):
        super().__init__()
        self.parent_alg = parent_alg
        self.ALG = 'exactGP'
        self.tag = '(GP) ----->'
        self.caution_tag = '(GP) !!!!!>'
        self.ENV_CONFIG = params_dict['environment']
        self.params_dict = params_dict
        self.num_sample = dataset['num_sample']

        self.x_dim = int(params_dict['observation_dim'])  # Observation dim = 10 in double ring.
        # self.x_dim = 4
        self.y_dim = int(params_dict['action_dim'])

        # Initialise Paths
        if os.path.isdir('C:/Amornyos/PhD/resources/T-ITS/training_records'):
            print(f'{self.tag}', 'This is my PC. Saved the training records to a specific loaction.')
            self.training_record_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/{:s}_{:s}_gp_training_record'.format(self.ALG, self.ENV_CONFIG)
            self.online_training_record_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/{:s}_{:s}_gp_online_training_record'.format(self.ALG, self.ENV_CONFIG)
            self.training_checkpoint_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/{:s}_{:s}_gp_training_record_checkpoint'.format(self.ALG, self.ENV_CONFIG)
            self.evaluation_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/{:s}_{:s}_evaluation_result'.format(self.ALG, self.ENV_CONFIG)
        else:
            print(f'{self.caution_tag}', 'This is not my PC. Saved the training record to the desktop.')
            import pathlib
            DESKTOP_PATH = pathlib.Path.home() / 'Desktop'
            self.training_record_path = f'{DESKTOP_PATH}/SGDP/{self.ALG}_{self.ENV_CONFIG}_gp_training_record'
            self.online_training_record_path = f'{DESKTOP_PATH}/SGDP/{self.ALG}_{self.ENV_CONFIG}_gp_online_training_record'
            self.training_checkpoint_path = f'{DESKTOP_PATH}/SGDP/{self.ALG}_{self.ENV_CONFIG}_gp_training_record_checkpoint'
            self.evaluation_path = f'{DESKTOP_PATH}/SGDP/{self.ALG}_{self.ENV_CONFIG}_evaluation_results'

        self.cuda() if cuda else ...

        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
            print(f'{self.tag} There exists a training record for this agent. Do you wish to load the exist one ?')
            _ans_1 = input(f'{self.tag} Press [y/n] and enter: ')
        else:
            _ans_1 = 'n'

        # # Check for the training record file and the response from user...
        # if load:
        #     _ans_1 = 'y'
        #     if not os.path.isfile(self.training_record_path) is True:
        #         print(f'{self.tag} There is no exists a training record for this agent !!')
        # else: 
        #     _ans_1 = 'n'

        # Check for the answer...
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print(f'{self.tag} Create new GP record and models!')
            self.mll_append = []  # Loss storage
            self.training_record = 0  # Training record counter

            # Initialise training data
            _start = 0
            _slice_idx = np.arange(stop=self.num_sample, step=10)
            self.x_train_original = torch.tensor(dataset['observations'][_slice_idx], dtype=torch.float32)
            self.x_train = torch.tensor(dataset['observations'][_slice_idx, :self.x_dim], dtype=torch.float32)
            self.y_train = torch.tensor(dataset['actions'][_slice_idx], dtype=torch.float32)

            # Create hyperparameters
            self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            self.ell = torch.nn.Parameter(torch.ones(size=[1, self.x_dim], dtype=torch.float32), requires_grad=True)
            # self.ell = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)

        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print(f'{self.tag} Load training record.')
            # Load training records
            _training_records = torch.load(self.training_record_path if not online else self.online_training_record_path, 
                                           weights_only=False, map_location=torch.device('cuda' if cuda else 'cpu'))
            self.mll_append = _training_records['loss_append']  # Loss append
            self.training_record = _training_records['training_record']  # Training record counter
            _state_dict = _training_records['state_dict']
            self.x_train = _training_records['x_train']
            self.y_train = _training_records['y_train']
            self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
            self.ell = torch.nn.Parameter(_state_dict['ell'], requires_grad=True)
            self.eval()
        else:
            print(f'{self.caution_tag} Try another answer. ("y" for yes (create new) or "n" for no (load existing one)).'.format(self.ALG))
            exit()

        # Initialised training kernels (K, K_inv) (For evaluation).
        self.sigma_p = torch.tensor(1.0, dtype=torch.float32)  # Fixed signal variance to 1.00^2
        self.K_mm = self.rbfKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)
        # print(self.y_train)
        self.num_sample = len(self.x_train)
        print(f'{self.tag} GP Initialised!')
        print(f'{self.tag} Algorithm: ', self.ALG)
        print(f'{self.tag} Num_Training_Sample: ', self.num_sample)
        print(f'{self.tag} Training_Record: ', self.training_record)

    def prepareTempData(self, temp_states, temp_actions):
        _num_sample = len(temp_states)
        _slice_idx = np.arange(stop=_num_sample, step=5)
        # self.x_train_temp = torch.tensor(temp_states[_slice_idx])
        # self.y_train_temp = torch.tensor(temp_actions[_slice_idx])
        # self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        # self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)
        self.x_train = torch.tensor(temp_states[_slice_idx], dtype=torch.float32)
        self.y_train = torch.tensor(temp_actions[_slice_idx], dtype=torch.float32)
        # Create hyperparameters
        self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.ell = torch.nn.Parameter(torch.ones(size=[1, self.x_dim], dtype=torch.float32), requires_grad=True)

    def kernelPreparation(self):
        self.K_mm = self.rbfKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

    def rbfKernel(self, X_1, X_2, noise=False):
        '''
        Isotropic squared exponential kernel.
        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).
        Returns:
            (m x n) matrix.
        '''
        X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2
        # print(X_1.size(), self.ell.size())
        kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1/self.ell, X_2/self.ell)**2)/2)
        # kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1, X_2)**2)/(2*self.ell))
        # kernel = torch.abs(kernel)
        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        return kernel

    def predict(self, X_s):
        with torch.no_grad():
            x_test = X_s[:, :self.x_dim]
            x_test = x_test.view(-1, self.x_dim)
            _k_s = self.rbfKernel(X_1=self.x_train, X_2=x_test, noise=False)
            _k_ss = self.rbfKernel(X_1=x_test, X_2=x_test, noise=False)

            # # Cholesky decomposition
            # _L = torch.linalg.cholesky(self.K_mm, upper=False)
            # _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, self.y_train, upper=False), upper=True)
            # _mean = _k_s.T @ _alpha
            # _v = torch.linalg.solve_triangular(_L, _k_s, upper=False)
            # _var = _k_ss - (_v.T @ _v)

            # Cholesky decomposition
            # _L = torch.linalg.cholesky(self.K_mm, upper=False)
            _alpha = torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, self.y_train, upper=False), upper=True)
            _mean = _k_s.T @ _alpha
            _v = torch.linalg.solve_triangular(self.L_mm, _k_s, upper=False)
            _var = _k_ss - (_v.T @ _v)

        return _mean, _var
    
    def myTraining(self, total_epoch:int, s_print:bool=False, online:bool=False):
        _batch_size = self.num_sample
        _gradient_step = int(len(self.x_train)//_batch_size)
        _cov_optimizer = torch.optim.Adam([self.sigma_n, self.ell], lr=1e-02 if not online else 1e-02)
        _training_record = 0
        _elbo_append = []

        while _training_record < total_epoch:
        # while self.training_record < total_epoch:
            _start_time = time()
            _sampling_indexes = torch.randperm(len(self.x_train))
            for g in range(_gradient_step):
                _batch_x = self.x_train[_sampling_indexes[g*_batch_size:_batch_size+(g*_batch_size)]]
                _batch_y = self.y_train[_sampling_indexes[g*_batch_size:_batch_size+(g*_batch_size)]]

                _cov_optimizer.zero_grad()
                _k = self.rbfKernel(X_1=_batch_x, X_2=_batch_x, noise=True)
                _L = torch.linalg.cholesky(_k, upper=False)
                _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, _batch_y, upper=False), upper=True)
                _mll = (-0.5 * _batch_y.T @ _alpha) - torch.sum(torch.log(torch.diagonal(_L))) - (_batch_size*torch.log(torch.tensor(2*torch.pi))/2)
                _mll = -_mll.mean()
                _mll.backward(retain_graph=True)
                _cov_optimizer.step()

                self.mll_append.append(_mll.tolist())

            self.training_record += 1
            _training_record += 1
        
            if s_print:
                print('Gradient_Step: ', _training_record, 
                    ', Loss: ', round(_mll.tolist(), 4), 
                    ', Obs_var: ', round(self.sigma_n.tolist(), 4),
                    ', Time_usage: ', round(time()-_start_time, 4))

            self.recordSaving(path=self.training_record_path if not online else self.online_training_record_path)

        self.training_record += _training_record
        self.recordSaving(path=self.training_record_path if not online else self.online_training_record_path)
        self.K_mm = self.rbfKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

        return np.mean(_elbo_append)

    def recordSaving(self, path:str):
        torch.save({'state_dict': self.state_dict(), 
                    'training_record': self.training_record,
                    'loss_append': self.mll_append,
                    'x_train': self.x_train, 
                    'y_train': self.y_train,
                    }, path)