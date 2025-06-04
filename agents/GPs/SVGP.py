import os
import torch
import numpy as np
from time import time

class mySVGP(torch.nn.Module):
    def __init__(self, parent_alg:str, params_dict:dict, dataset:dict, cuda:bool=False):
        super().__init__()
        self.parent_alg = parent_alg
        self.ALG = 'svgp'
        self.tag = '(GP) ----->'
        self.caution_tag = '(GP) !!!!!>'
        self.ENV_CONFIG = params_dict['environment']
        self.params_dict = params_dict
        # self.num_sample = params_dict['gp_num_sample']
        # self.num_inducing = params_dict['gp_num_inducing']
        self.num_sample = int(params_dict['horizon'] / 10)
        self.num_inducing = 256
        self.x_dim = int(params_dict['observation_dim'])
        self.y_dim = int(params_dict['action_dim'])

        # Initialise Paths
        if os.path.isdir('C:/Amornyos/PhD/resources/T-ITS/training_records'):
            print(f'{self.tag}', 'This is my PC. Saved the training records to a specific loaction.')
            self.training_record_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/{:s}_{:s}_training_record'.format(self.ALG, self.ENV_CONFIG)
            self.training_checkpoint_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/{:s}_{:s}_training_record'.format(self.ALG, self.ENV_CONFIG)
            self.evaluation_path = 'C:/Amornyos/PhD/resources/T-ITS/training_records/{:s}_{:s}_evaluation_result'.format(self.ALG, self.ENV_CONFIG)
        else:
            print(f'{self.caution_tag}', 'This is not my PC. Saved the training record to the desktop.')
            import pathlib
            DESKTOP_PATH = pathlib.Path.home() / 'Desktop'
            self.training_record_path = f'{DESKTOP_PATH}/{self.ALG}_{self.ENV_CONFIG}_training_record'
            self.training_checkpoint_path = f'{DESKTOP_PATH}/{self.ALG}_{self.ENV_CONFIG}_training_record_checkpoint'
            self.evaluation_path = f'{DESKTOP_PATH}/{self.ALG}_{self.ENV_CONFIG}_evaluation_results'

        self.cuda() if cuda else ...

        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.ALG))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: '.format(self.ALG))
        else:
            _ans_1 = 'n'
        # Check for the answer...
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== ({:s}) Create new GP record and models!'.format(self.ALG))
            self.mll_append = []  # Loss storage
            self.training_record = 0  # Training record counter
            # Initialise training data
            _start = 0
            self.x_train = torch.tensor(dataset['observations'][0+_start:_start+self.num_sample], dtype=torch.float32)
            self.y_train = torch.tensor(dataset['actions'][0+_start:_start+self.num_sample], dtype=torch.float32)
            # Intialise inducing points (Let's fix it first.)
            self.z_train, self.indexes = self.getInitInducingPoints(x_train=self.x_train)
            # Create hyperparameters
            self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            self.ell = torch.nn.Parameter(torch.ones(size=[1, self.x_dim], dtype=torch.float32), requires_grad=True)
            self.q_mean = torch.nn.Parameter(self.y_train[self.indexes], requires_grad=True)
            # self.q_var = torch.nn.Parameter(torch.eye(self.num_inducing), requires_grad=True)  # Single covariance
            self.q_var = torch.nn.Parameter(torch.ones([self.num_inducing, self.num_inducing]), requires_grad=True)  # Single covariance

        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== ({:s}) Load training record.'.format(self.ALG))
            # Load training records
            _training_records = torch.load(self.training_record_path, weights_only=False, map_location=torch.device('cuda' if cuda else 'cpu'))
            self.mll_append = _training_records['gp_loss_append']  # Loss append
            self.training_record = _training_records['gp_training_record']  # Training record counter
            _state_dict = _training_records['gp_state_dict']
            self.x_train = _training_records['gp_x_train']
            self.y_train = _training_records['gp_y_train']
            self.z_train = _training_records['gp_z_train']
            self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
            self.ell = torch.nn.Parameter(_state_dict['ell'], requires_grad=True)
            self.q_mean = torch.nn.Parameter(_state_dict['q_mean'], requires_grad=True)
            self.q_var = torch.nn.Parameter(_state_dict['q_var'], requires_grad=True)
            self.eval()
        else:
            print('========== ({:s}) Try another answer. ("y" for yes (create new) or "n" for no (load existing one)).'.format(self.ALG))
            exit()

        # Initialised training kernels (K, K_inv) (For evaluation).
        self.sigma_p = torch.tensor(1.0, dtype=torch.float32)  # Fixed signal variance to 1.00^2
        self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

        print(f'{self.tag}', 'GP Initialised !')
        print(f'{self.tag}', 'Num_Training_Sample: ', self.num_sample)
        print(f'{self.tag}', 'Num_Inducing: ', self.num_inducing)
        print(f'{self.tag} sigma_n: {self.sigma_n}')
        print(f'{self.tag} ell: {self.ell}')

    def getInitInducingPoints(self, x_train):
        _indexes = np.linspace(start=0, stop=len(x_train), num=self.num_inducing, endpoint=False)
        return x_train[_indexes], _indexes

    def kernelPreparation(self):
        self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
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
        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        return kernel

    def predict(self, X_s):
        with torch.no_grad():
            x_test = X_s.view(-1, self.x_dim)
            _k_s = self.rbfKernel(X_1=self.z_train, X_2=x_test, noise=False)
            _k_ss = self.rbfKernel(X_1=x_test, X_2=x_test, noise=False)

            # # We found that deriving mean and variance this way, provide better results.
            # _mean = _k_s.T @ self.K_inv @ self.y_train
            # _var = _k_ss - _k_s.T @ self.K_inv @ _k_s

            _mean = _k_s.T @ (torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, self.q_mean, upper=False), upper=True))
            _v = torch.linalg.solve_triangular(self.L_mm, _k_s, upper=False)
            _k_tilde = _k_ss - (_v.T @ _v)
            _1 = torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, self.q_var, upper=False), upper=True)
            _2 = torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, _k_s, upper=False), upper=True)
            _sigma_f = _k_tilde + (_k_s.T @ _1 @ _2)
            _var = (_sigma_f) + (self.sigma_n**2)

        return _mean, _var

    def myTraining(self, total_epoch:int, ft:bool=False):
        # _batch_size = self.num_sample if not ft else self.x_train.size()[0]  # minibatch size is the whole dataset.
        _batch_size = self.num_inducing
        _gradient_step = int(self.num_sample//_batch_size)  
        _cov_optimizer = torch.optim.Adam([self.sigma_n, self.ell], lr=1e-02)
        _ind_optimizer = torch.optim.Adam([self.q_mean, self.q_var, self.z_train], lr=1e-02)
        _training_record = 0
        _elbo_append = []

        while _training_record < total_epoch:
            _start_time = time()
            for g in range(_gradient_step):
                _batch_x = self.x_train[g*_batch_size:_batch_size+(g*_batch_size)]
                _batch_y = self.y_train[g*_batch_size:_batch_size+(g*_batch_size)]
                _batch_z = self.z_train
                _l_1 = 0
                _cov_optimizer.zero_grad()
                _ind_optimizer.zero_grad()
                _K_mm = self.rbfKernel(X_1=_batch_z, X_2=_batch_z, noise=True)
                _K_nn = self.rbfKernel(X_1=_batch_x, X_2=_batch_x, noise=True)
                # _K_mm_inv = torch.linalg.inv(_K_mm)
                _L_mm = torch.linalg.cholesky(_K_mm, upper=False)
                _K_mn = self.rbfKernel(X_1=_batch_z, X_2=_batch_x, noise=False)
                _K_tilde = _K_nn - _K_mn.T @ (torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, _K_mn, upper=False), upper=True))

                # _cov = (self.sigma_n**2) * torch.eye(self.y_dim)  # Inverse precision
                # _L_sigma = torch.linalg.cholesky(_cov, upper=False)
                for h in range(_batch_size-1):
                    _k_i = _K_mn[:, h].view(-1, 1)
                    _mean = _k_i.T @ (torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, self.q_mean, upper=False), upper=True))
                    
                    # _alpha = torch.linalg.solve_triangular(_L_sigma.T, torch.linalg.solve_triangular(_L_sigma, (_batch_y[h].view(-1, self.y_dim)-_mean), upper=False), upper=True)
                    # _cap_lambda = ((_K_mm_inv @ _k_i) @ (_k_i.T @ _K_mm_inv)) / (self.sigma_n**2)
                    # _l_1 += (-0.5 * (_batch_y[h]-_mean) @ _alpha) - torch.sum(torch.diagonal(_L_sigma)) - \
                    #         (torch.log(torch.tensor(2*torch.pi))/2) - \
                    #         ((0.5 * _K_tilde[h, h])/(self.sigma_n**2)) - (0.5*torch.sum(torch.diagonal(self.q_var @ _cap_lambda)))

                    _1 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, self.q_var, upper=False), upper=True)
                    _2 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, _k_i, upper=False), upper=True)
                    _sigma_f = torch.abs(_K_tilde[h, h].view(1, 1) + (_k_i.T @ _1 @ _2))

                    _l_1 += (-(_batch_y[h].view(1, self.y_dim)-_mean)**2) / (2*(self.sigma_n**2)) - (torch.log(torch.sqrt(2*torch.pi*(self.sigma_n**2)))) - \
                            (_sigma_f / (2*(self.sigma_n**2)))

                # KL-divergence term
                _cov_q = torch.sqrt(torch.abs(torch.diagonal(self.q_var))) @ torch.ones(size=[self.num_inducing, self.y_dim])
                _q = torch.distributions.Normal(loc=self.q_mean, scale=_cov_q)
                _u = _q.sample(torch.Size())
                _log_q_u = _q.log_prob(_u)
                _p = torch.distributions.Normal(loc=0., scale=torch.sqrt(torch.abs(torch.diagonal(_K_mm)))@torch.ones(size=[self.num_inducing, self.y_dim]))

                # ELBO
                _elbo = _l_1 - (-torch.sum(torch.exp(_log_q_u)*(_p.log_prob(_u)-_log_q_u)))
                _elbo = -_elbo.mean()
                _elbo.backward(retain_graph=True)
                _cov_optimizer.step()
                _ind_optimizer.step()

                _elbo_append.append(_elbo.tolist())
                self.mll_append.append(_elbo.tolist())

            _training_record += 1
        
            print('Gradient_Step: ', _training_record, 
                  ', Loss: ', round(_elbo.tolist(), 4), 
                  ', Var: ', round(torch.diagonal(self.q_var).mean().tolist(), 4),
                  ', Obs_var: ', round(self.sigma_n.tolist(), 4),
                  ', Time_usage: ', round(time()-_start_time, 4))

        self.training_record += _training_record
        # self.recordSaving(path=self.training_record_path)
        self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

        return np.mean(_elbo_append)
    
    def recordSaving(self, path:str):
        torch.save({'state_dict': self.state_dict(), 
                    'training_record': self.training_record,
                    'loss_append': self.mll_append,
                    'x_train': self.x_train, 
                    'y_train': self.y_train,
                    'z_train': self.z_train}, path)