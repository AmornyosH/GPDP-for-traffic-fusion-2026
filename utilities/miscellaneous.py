
import numpy as np
import torch

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

# def flatten1D(input):
#     '''
#     Variables flatten method (Convert N-D arrays to 1-D list) \\
#     This method will convert any input in N-D arrays format to 1-D (list) format.
#     This method can receive 'tensor' and 'numpy' format.

#     Input Arguments: input:Any
#     Return: flatten_input (Numpy Format)
#     '''
#     if tf.is_tensor(input):
#         flatten_input = tf.concat([tf.reshape(item, [-1]) for item in input], axis=0)
#         # Convert to numpy
#         flatten_input = flatten_input.numpy()
#     else:
#         flatten_input = np.concatenate([tf.reshape(item, [-1]) for item in input], axis=0)
#     return  flatten_input

# def assignVarsToModels(model, input):
#     '''
#     Variable assignment method \\
#     This method will assign the value from input list (1-D) to NNs model. \\
#     Usually, this method is created for assigned new weights computed from optimisation problem \\
#     to the weigths of neural networks. \\
    
#     Input Arguments: model(NN), input:(1-D list)
#     Returns: model after new weigths assigned.
#     '''
#     shapes = [item.shape.as_list() for item in model.trainable_variables]
#     size_theta = np.sum([np.prod(item) for item in shapes])
#     start = 0
#     for idx, item in enumerate(shapes):
#         size = np.prod(item)
#         param = tf.reshape(input[start:start + size], item)
#         model.trainable_variables[idx].assign(param)
#         start += size
#     assert start == size_theta, 'messy shapes'
#     return model

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