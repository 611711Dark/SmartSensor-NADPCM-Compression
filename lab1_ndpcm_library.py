
from collections import namedtuple
import numpy as np

# Import quantizer for error
import lab1_library

# Declaring namedtuple()
# n - total length of the simulation (number of samples/iterations)
# h_depth - number of history elements in \phi and corresponding coefficients (length of vectors)
# n_bits - number of bits to be transmitted (resolution of encoded error value)
# phi - vector of vectors of samples history (reproduced!!) 
#       - first index = iteration; second index = current time vector element
# theta - vector of vectors of coefficients 
#       - first index = iteration; second index = current time vector element
# y_hat - vector of all predicted (from = theta * phi + k_v * eq)
# e - exact error between the sample and the predicted value (y_hat)
# eq - quantized value of error (see n_bits!!)
# y_recreated - vector of all recreated/regenerated samples (used in the prediction!!)
NDPCM = namedtuple('NDAPCM', ['n', 'h_depth', 'n_bits',
                   'phi', 'theta', 'y_hat', 'e', 'eq', 'y_recreated'])
alpha = 0.002
k_v = 0.002

def init(n, h_depth, n_bits):
    # Adding values
    data_block = NDPCM(
        n, 
        h_depth, 
        n_bits, 
        np.zeros((n, h_depth)), 
        np.zeros((n, h_depth)), 
        np.zeros(n), 
        np.zeros(n), 
        np.zeros(n), 
        np.zeros(n)
    )
    data_block.phi[0] = np.zeros(h_depth)
    data_block.theta[0] = np.zeros(h_depth)
    data_block.y_recreated[0] = 0
    ### Modify initial value for any component, parameter:
    # ...
    return data_block

def calculate_alpha_max(phi):
    phi_norm_sq = np.dot(phi, phi)
    if phi_norm_sq < 1e-10:
        result = 0.002  
    else:
        result = 1 / phi_norm_sq
    return result

def calculate_kv_max(phi, alpha_):
    phi_norm_sq = np.dot(phi, phi)
    
    stability_term = 1 - alpha_ * phi_norm_sq
    if stability_term <= 1e-10:  
        stability_term = 1e-10
    
    kv_max = np.sqrt(stability_term)
    return kv_max

def prepare_params_for_prediction(data_bloc, k):
    # Update weights for next round (k) based on previous k-1, k-2,...
    # TODO: for first iteration INITIALIZE 'phi' and 'theta'
    if (k == 1):
        data_bloc.phi[0] = np.zeros(data_bloc.h_depth)
        data_bloc.theta[0] = np.zeros(data_bloc.h_depth)
        return
    
    # TODO: Fill 'phi' history for 'h_depth' last elements
    for i in range(0, data_bloc.h_depth-1):
        data_bloc.phi[k-1][i] = data_bloc.y_recreated[k-i-1]

    alpha = calculate_alpha_max(data_bloc.phi[k-1])
    # print("e=", data_bloc.eq[k])
    # print("eT=", data_bloc.eq[k].transpose())
    # TODO: Update weights/coefficients 'theta'

    data_bloc.theta[k-1] = data_bloc.theta[k-2] + alpha * data_bloc.eq[k-1] * data_bloc.phi[k-2]

    return

def predict(data_bloc, k):
    
    k_v=calculate_kv_max(data_bloc.phi[k-1], alpha)

        # Initial prediction - use first sample value or zero
    if k == 0:
        data_bloc.y_hat[k] = data_bloc.y_recreated[0] if k == 0 else 0
    
        # Main prediction equation:
        # ŷ(k) = θ(k-1)•φ(k-1) - k_v * eq(k-1)
    #if (k > 0):
    #    data_bloc.phi[k] = data_bloc.phi[k-1]
    # TODO: calculate 'hat y(k)' based on (k-1) parameters
    else:
        data_bloc.y_hat[k] = np.dot(data_bloc.theta[k-1], data_bloc.phi[k-1]) - k_v * data_bloc.eq[k-1]
    # data_block.y_hat[k] = ...
    # if (k==1):
        # data_block.y_hat[k] = ...
    # print ( data_bloc.theta[k] @ data_bloc.phi[k])
    # TODO: Return prediction - fix:
    # return data_bloc.y_recreated[k-1];
    return data_bloc.y_hat[k]


def calculate_error(data_block, k, real_y):
    data_block.e[k] = real_y - data_block.y_hat[k]
    data_block.eq[k] = lab1_library.quantize(data_block.e[k], data_block.n_bits)
    return data_block.eq[k]


def reconstruct(data_block, k):
    data_block.y_recreated[k] = data_block.y_hat[k] + data_block.eq[k]
    return data_block.y_recreated[k]
