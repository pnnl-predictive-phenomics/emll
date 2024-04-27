
import pymc as pm
import pytensor 
import pandas  as pd
import numpy as np
import pytensor.tensor 
import pytensor.tensor as pt
from typing import Optional

# test create_pytensor_from_data_naive == create_pytensor_from_data_fancy
# test create_pytensor_from_data_naive == initialize_elasticity


# reactions x number of conditions 
# pymc tensor for number of variables x number of conditions 
# for observed values -> instead of normal()  


# Test cases: compare dictionaries using 'mixed dataframe'. 


def create_noisy_observations_of_computed_values(name:str, computed_tensor:pt.tensor,data:pd.DataFrame)->dict[tuple[int,int],pt.tensor]:
    """"""
    # check shape of computed tensor = data

    rv = dict()
    # create noisy observations of computed values
    for i in range(N):
        row_elements = []
        for j in range(M):
            rv[i,j] = pm.Normal.dist(name='{name}_{i}_{j}_obs', mu=np.log(computed_tensor[i,j]), sigma=np.sqrt(np.abs(np.log(data.loc[i,j]))), shape=(1,), observed=data.loc[i,j])
    return rv


def create_informed_priors_from_data_naive(name:str, data:pd.DataFrame)->pt.tensor:
    """Creates an NxM pytensor from a dataframe of N conditions (rows) and M variables (columns)."""

    # Performance: ~12 seconds for 10,000 elements, ~1200 seconds (20 min) for 1,000,000 elements

    # Todo: track indices for unit test

    # parameters from distributions (ij+distribution_type, parameters)
    # dataframe = N model variables (col) x M measured conditions (rows)
    # Inf (unmeasured) | NaN (zero/excluded) | float (measured)

    df = data
    N, M = df.shape  # N rows (conditions), M columns (variables)

    # check tensor shape == df shape

    tensor_elements = []
    obs_index = []
    unobs_index = []
    zeros_index = []
    index_dictionary = {}

    index_count = 0

    for i in range(N):
        row_elements = []
        for j in range(M):
            
            element = df.iloc[i, j]
            if np.isfinite(element):  
                # finite --> observed model variable --> Normal(log(variable value), 0.2)
                assert element >0, "{element} must be > 0 for log_e(x)"
                rv = pm.Normal.dist(name='{name}_{i}_{j}', mu=np.log(element), sigma=0.2, shape=(1,))
                row_elements.append(rv)
                obs_index.append(index_count)
                print(f'element {i},{j} = col:{df.columns[j]}, row:{df.index[i]} = {element}. --> Normal(log({element}),0.2)')

            elif np.isinf(element):
                # Inf --> unobserved model variable --> Laplace(0,0.1)
                rv = pm.Laplace.dist(name='{name}_{i}_{j}', mu=0, b=0.1, shape=(1,))
                row_elements.append(rv)
                unobs_index.append(index_count)
                print(f'element {i},{j} = col:{df.columns[j]}, row:{df.index[i]} = {element}. --> Laplace(0,0.1)')
            elif np.isnan(element):
                # inf --> model variable is zero (excluded)
                row_elements.append(pt.zeros(1))
                zeros_index.append(index_count)
                print(f'element {i},{j} = col:{df.columns[j]}, row:{df.index[i]} = {element}. --> 0')
            index_count += 1
    
        tensor_elements.append(row_elements)

    # Stack the elements into a 2D tensor
    data_tensor = pt.stack(tensor_elements, axis=0)

    index_dictionary['observed'] = obs_index
    index_dictionary['unobserved'] = unobs_index
    index_dictionary['zeros'] = zeros_index
    return data_tensor, index_dictionary


def create_pytensor_from_data_fancy(data_params:dict, model_context=True):
    """Creates a pytensor based on data - including missing model variables and experimental conditions."""
    # Create a random stream with a fixed seed
    seed = 1
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    if model_context:
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
    else:
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



if __name__ == "__main__":

    # Simple test data sets
    all_observed_dict = {
        'x':[3.0, 2.0, 1.0],
        'y':[1.0, 2.0, 3.0],
        'z':[1.0, 1.0, 1.0],
    }

    all_unobserved_dict = {
        'x':[np.inf, np.inf, np.inf],
        'y':[np.inf, np.inf, np.inf],
        'z':[np.inf, np.inf, np.inf],
    }

    all_zeros_dict = {
        'x':[np.nan, np.nan, np.nan],
        'y':[np.nan, np.nan, np.nan],
        'z':[np.nan, np.nan, np.nan],
    }

    mixed_columns_dict = {
        'x':[1.0, 2.0, 3.0],
        'y':[np.inf, np.inf, np.inf],
        'z':[np.nan, np.nan, np.nan],
    }

    mixed_rows_dict = {
        'x':[1.0, np.inf, np.nan],
        'y':[2.0, np.inf, np.nan],
        'z':[3.0, np.inf, np.nan],
    }

    fully_mixed_dict = {
        'x':[np.Inf, np.NaN, 1.0],
        'y':[np.NaN, 2.0, np.Inf],
        'z':[0.1, np.Inf, np.NaN],
    }



    test_dicts = [all_observed_dict,all_unobserved_dict,all_zeros_dict,mixed_columns_dict,mixed_rows_dict,fully_mixed_dict]

    for test_dict in test_dicts:
        print("\nINPUT:")
        test_df = pd.DataFrame.from_dict(test_dict)
        print(test_df)
        print("OUTPUT")
        simple_tensor, index_dictionary = create_pytensor_from_data_naive(test_df)


    # time using larger dataframe
    import time
    # move to test_data folder
    enzyme_fname = '/Users/geor228/Library/CloudStorage/OneDrive-PNNL/Desktop/prepared_enzyme_activity_data.csv'

    enzyme_df = pd.read_csv(enzyme_fname, index_col=0)
    print(enzyme_df.head())
    t0 = time.time()
    enzyme_tensor, index_dictionary = create_pytensor_from_data_naive(enzyme_df)

    print(f"{time.time()-t0} seconds")
    # # Define the dimensions of the DataFrame
    # N = 1000
    # M = 27

    # # Generate a random NumPy array with the given dimensions
    # random_array = np.random.randn(N, M)  # Standard normal distribution

    # # Define the proportion of NaNs and Infs you want
    # prop_nan = 0.33  
    # prop_inf = 0.33 

    # # Randomly choose indices for NaNs and Infs
    # nan_indices = np.random.choice(N * M, size=int(N * M * prop_nan), replace=False)
    # inf_indices = np.random.choice(N * M, size=int(N * M * prop_inf), replace=False)

    # # Flatten the array to make indexing easier
    # flat_random_array = random_array.flatten()

    # # Set NaNs and Infs
    # flat_random_array[nan_indices] = np.nan
    # flat_random_array[inf_indices] = np.inf

    # # Reshape the array back to its original shape
    # random_array = flat_random_array.reshape(N, M)

    # # Create a pandas DataFrame from the NumPy array
    # random_df = pd.DataFrame(random_array)
    # t0 = time.time()
    # random_tensor = create_pytensor_from_data_naive(random_df)
    # t1 = time.time()
    # total = t1-t0
    # print(f"total time for {N*M} elements: {total} s")