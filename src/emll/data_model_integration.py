
import pymc as pm
import pytensor 
import pandas 
import numpy as np
import pytensor.tensor 


def get_data_params_from_data(data):
    data_params = {}
    return data_params

def create_pytensor_from_data(data_params:dict):
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
        
        data_excluded_tensor = pytensor.tensor.zeros(data_params['excluded_shape'])
    
        data_tensor = pytensor.tensor.concatenate(
            [data_measured_dist, 
            data_unmeasured_dist, 
            data_excluded_tensor,
            ], axis=1
            )[:, data_params['tensor_indexer']]

    return data_tensor