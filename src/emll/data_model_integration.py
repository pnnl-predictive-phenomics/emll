
import pymc as pm
import pytensor 
import pandas 
import numpy as np
import pytensor.tensor 


# test create_pytensor_from_data_naive == create_pytensor_from_data_fancy
# test create_pytensor_from_data_naive == initialize_elasticity


def create_pytensor_from_data_naive(data):

    # parameters from distributions (ij+distribution_type, parameters)
    # dataframe = N model variables x M measured conditions
    # NaN (unmeasured) | Inf (zero/excluded) | float (measured)
    # create empty lists for unmeasured, measured, and excluded indices
    # for each row i
    #   for each column j
    #       if df[ij] == Nan
    #           add ij (index) to unmeasured list
    #           create distribution (what is the shape of zero dimensional object)
    #       if df[ij] == Inf
    #           add ij (index) to excluded list
    #           create distribution (what is the shape of zero dimensional object)
    #       if df[ij] == float
    #           add ij (index) to measured list
    #           create distribution (what is the shape of zero dimensional object)
    # return the new tensor
    raise NotImplementedError

def create_pytensor_from_data_fancy(data_params:dict):
    """Creates a pytensor based on data - including missing model variables and experimental conditions."""
    # Create a random stream with a fixed seed
    seed = 1
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    with pm.Model() as pymc_model:
        data_measured_dist = pm.Normal(data_params['measured_name'], 
                            mu=data_params['measured_mu'], 
                            sigma=data_params['measured_sigma'], 
                            shape=(data_params['measured_shape']),
                            rng=rng
                            )

        data_unmeasured_dist = pm.Laplace(data_params['unmeasured_name'], 
                                 mu=data_params['unmeasured_mu'], 
                                 b=data_params['unmeasured_b'], 
                                 shape=(data_params['unmeasured_shape']),
                                 rng=rng
                                 )
        
        data_zero_tensor = pytensor.tensor.zeros(data_params['excluded_shape'])
    
        data_tensor = pytensor.tensor.concatenate(
            [data_measured_dist, 
            data_unmeasured_dist, 
            data_zero_tensor,
            ], axis=1
            )[:, data_params['tensor_indexer']]

    return data_tensor