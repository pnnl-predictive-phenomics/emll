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

    # check that model context exists
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


def create_pytensor_from_data_naive(name:str, data:pd.DataFrame, normal_stdev:pd.DataFrame, laplace_loc_and_scale:pd.date_range)->T.tensor:  # noqa: D417
    """Creates a pytensor from data with missing values. 

    Args:
        name (str): Name to use for random variables (i.e., {name}_{row}_{col}).
        data (pd.DataFrame): Data with rows = n conditions, columns = m model variables. 
        Each value must be np.finite for observed, np.inf for unobserved, np.nan for excluded.
        normal_stdev (pd.DataFrame): Standard deviations to use for observed data. 
        Each value must be either np.nan or positive np.finite, and the dataframe must have the 
        same shape as the data.
        laplace_loc_and_scale (pd.date_range): Laplace location (mu) and scale (b) parameters for 
        unobserved variables. Each value must be either a tuple (mu,b) or np.nan, with mu np.finite
        and b positive finite, and dataframe must have the same shape as the data. 

    Returns:
        T.tensor: Stacked pytensor of normal dist, laplace dist, and zeros.
    """

    # Check input types
    if not isinstance(name, str):
        raise TypeError(f"Expected name to be of type {str.__name__}, but got {type(str).__name__} instead.")
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected data to be of type {pd.DataFrame.__name__}, but got {type(data).__name__} instead.")

    if not isinstance(normal_stdev, pd.DataFrame):
        raise TypeError(f"Expected normal_stdev to be of type {pd.DataFrame.__name__}, but got {type(normal_stdev).__name__} instead.")

    if not isinstance(laplace_loc_and_scale, pd.DataFrame):
        raise TypeError(f"Expected laplace_loc_and_scale to be of type {pd.DataFrame.__name__}, but got {type(laplace_loc_and_scale).__name__} instead.")


    # Check input shapes match
    if data.shape != normal_stdev.shape:
            raise ValueError(f"Data shape {data.shape} does not match the standard deviation shape {normal_stdev.shape}!")

    if data.shape != laplace_loc_and_scale.shape:
            raise ValueError(f"Data shape {data.shape} does not match the laplace loc and scale shape {laplace_loc_and_scale.shape}!")

    # Function to check if the data value is finite, np.nan, or np.inf
    def check_data_value(x):
        if isinstance(x, (int, float)):
            return np.isfinite(x) or np.isnan(x) or np.isinf(x)
        else:
            return False

    # Function to check if the stdev value is np.nan or finite > 0.
    def check_stdev_value(x):
        # Check if x is a number (either int or float)
        if isinstance(x, (int, float)):
            # Check if it's np.nan
            if np.isnan(x):
                return True
            # Check for finite numbers > 0
            return np.isfinite(x) and x > 0
        # If it's not a number, return False
        return False
    
    def check_laplace_value(x):
        if isinstance(x, tuple):
            # Check that the tuple has only two items
            if len(x) != 2:
                return False
            # Check that the first item is finite
            if not np.isfinite(x[0]):
                return False
            # Check that the second item is positive and finite
            if not (np.isfinite(x[1]) and x[1] > 0):
                return False
            return True
        elif pd.isna(x):
            # The value is np.nan
            return True
        else:
            # The value is something else
            return False
    
    # Check that all values in the data DataFrame are finite, np.nan, or np.inf
    if not data.map(check_data_value).all().all():
        raise ValueError("DataFrame contains values that are not finite, np.nan, or np.inf.")
    
    # Check that all values in the stdev DataFrame are nan or finite > 0
    if not normal_stdev.map(check_stdev_value).all().all():
        raise ValueError("Stdev DataFrame contains invalid values. Valid values are np.nan or finite numbers > 0.")
   
    # Check that all values in the laplace DataFrame are tuple(finite, postive finite) or np.nan
    if not laplace_loc_and_scale.map(check_laplace_value).all().all():
        raise ValueError("Laplace DataFrame contains invalid values that are not tuple(finite, postive finite) or np.nan")

    # Check that the columns are the same for data and standard deviation
    if not data.columns.equals(normal_stdev.columns):
        raise ValueError("Data and standard deviations column labels are different.")

    # check that data and stdev rows match
    if not data.index.equals(normal_stdev.index):
        raise ValueError("The indices of data and standard deviation do not match.")
    
    # Check that the columns are the same for data and laplace
    if not data.columns.equals(laplace_loc_and_scale.columns):
        raise ValueError("Data and laplace column labels are different.")

    # Check that the rows are the same for data and laplace
    if not data.index.equals(laplace_loc_and_scale.index):
        raise ValueError("Data and laplace indices are different.")


    # check that model context exists
    try:
        model_context = pm.Model.get_context()
    except TypeError as e:
        raise TypeError("Function must be run within a PyMC model context.") from e

    tensor_elements = []
    for i, row in enumerate(data.index):
        row_elements = []
        for j, col in enumerate(data.columns):
            data_value = data.loc[row,col]
            if np.isfinite(data_value):  
                # finite --> observed model variable --> Normal(log(variable value), 0.2)
                sigma_value = normal_stdev.loc[row,col]
                rv = pm.Normal.dist(name=f'{name}_{row}_{col}', mu=data_value, sigma=sigma_value)
            elif np.isinf(data_value):
                # Inf --> unobserved model variable --> Laplace(loc,scale)
                loc_value = laplace_loc_and_scale.loc[row,col][0]
                scale_value = laplace_loc_and_scale.loc[row,col][1]
                rv = pm.Laplace.dist(name=f'{name}_{row}_{col}', mu=loc_value, b=scale_value)
            elif np.isnan(data_value):
                # Nan --> model variable is zero (excluded)
                rv = T.zeros(())  # Scalar zero tensor
            row_elements.append(rv)  # Scalar zero tensor
        row_tensor = T.stack(row_elements, axis=1)
        tensor_elements.append(row_tensor)
    data_tensor = T.stack(tensor_elements, axis=0)
    return data_tensor
