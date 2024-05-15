import pymc as pm
import pytensor 
import pandas as pd
import numpy as np
import pytensor.tensor as T

def create_noisy_observations_of_computed_values(name:str, computed_tensor:T.tensor, data:pd.DataFrame, estimated_stdev:pd.DataFrame)->dict[tuple[int,int], T.tensor]:  # noqa: D417
    """Create noisy observations of computed values.

    Parameters
    ----------
        name (str): The name of the computed tensor.
        computed_tensor (T.tensor): The computed tensor used for mu (mean).
        data (pd.DataFrame): The data used for observations.
        estimated_stdev (pd.DataFrame): The estimated standard deviation.

    Returns
    -------
        dict[tuple[int,int],pytensor.tensor]: A dictionary containing the noisy observations of computed values.

    """
    # check shape of data == estimated_stdev
    if data.shape != estimated_stdev.shape:
        raise ValueError("Data shape does not match standard deviation shape!")

    # check shape of data == computed_tensor
    if any(data.shape != computed_tensor.shape.eval()):
        raise ValueError(f"Data shape {data.shape} does not match computed tensor shape {pytensor.tensor.shape(computed_tensor)}!")

    # check that standard deviations are all positive
    if (estimated_stdev.values<0).any():
        raise ValueError("Estimated standard deviation has negative values!")
     
    # check that data and stdev rows match
    if not data.index.equals(estimated_stdev.index):
        raise ValueError("The indices of 'data' and 'estimated_stdev' do not match.")

    # check that data and stdev columns match
    if not data.columns.equals(estimated_stdev.columns):
        raise ValueError("Data and estimated standard deviations column labels are different.")

    try:
        model_context = pm.Model.get_context()
    except TypeError as e:
        raise ValueError("Function must be run within a PyMC model context.") from e

    rv = {}
    
    # create noisy observations of computed values
    for i, row in enumerate(data.index):
        for j, col in enumerate(data.columns):
            observed_data = data.loc[row, col]
            if np.isfinite(observed_data):
                mean_value = computed_tensor[i,j]
                stdev_value = estimated_stdev.loc[row,col]
                rv[row,col] = pm.Normal(name=f'{name}_{row}_{col}_obs',
                                             mu=mean_value,
                                             sigma=stdev_value,
                                             observed=observed_data)
    return rv


def create_pytensor_from_data(name:str, data:pd.DataFrame)->T.tensor:  # noqa: D417
    """"""

    # Check input types
    if not isinstance(name, str):
        raise TypeError(f"Expected name to be of type {str.__name__}, but got {type(str).__name__} instead.")
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected data to be of type {pd.DataFrame.__name__}, but got {type(data).__name__} instead.")

    # Function to check if a value is finite, np.nan, or np.inf
    def check_value(x):
        if isinstance(x, (int, float)):
            return np.isfinite(x) or np.isnan(x) or np.isinf(x)
        else:
            return False

    # Check that all values in the DataFrame are finite, np.nan, or np.inf
    if not data.map(check_value).all().all():
        raise ValueError("DataFrame contains values that are not finite, np.nan, or np.inf.")
    
    try:
        model_context = pm.Model.get_context()
    except TypeError as e:
        raise TypeError("Function must be run within a PyMC model context.") from e

