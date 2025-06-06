import numpy as np
import torch
from time import time

from agents.RLs.SGDP import StochasticGaussianprocessDiffusionPolicy
# from agents.RLs.SGDQ import StochasticGaussianprocessDiffusionQlearning

TAG = '(Offline Training) ----->'
CAUTION_TAG = '(Offline Training) !!!!!>'

def offlineTraining(path:str, env:object, alg:str, cont:bool, dataset:dict, total_steps:int):
    # Initialise agent's object
    if alg == 'SGDP':
        agent = StochasticGaussianprocessDiffusionPolicy(params_dict=env.params_dict, dataset=dataset, ft=False)

    # Initialise parameters
    _steps = agent.training_step_counter
    _total_steps = total_steps
    _epoch = agent.training_record
    _steps_per_epoch = int(dataset['num_sample']/agent.MINIBATCH_SIZE)
    _total_epochs = int(_total_steps/_steps_per_epoch)

    # Start training
    while _epoch < _total_epochs:
        _start_time = time()
        _training_results = agent.training(steps=_steps_per_epoch)
        _epoch += 1
        print(f'{TAG} Epoch: {_epoch}/{_total_epochs},',
              f' Gradient Step: {agent.training_step_counter}/{_total_steps}',
              ' P-Loss: ', round(_training_results['policy_loss'], 2),
              ' Q1-Loss: ', round(_training_results['q1_loss'], 2),
              ' Q2-Loss: ', round(_training_results['q2_loss'], 2) if _training_results.get('q2_loss') is not None else '',
              ' GP-Loss: ', round(_training_results['gp_loss'], 2) if _training_results.get('gp_loss') is not None else '',
              ' Time: ', round(time()-_start_time, 2)
              )
    # GP training.
    agent.gp_model.y_train = agent.getAlteredObservation(states=agent.gp_model.x_train)
    agent.gp_model.myTraining(total_epoch=200)
    agent.recordSaving(path=agent.training_record_path)