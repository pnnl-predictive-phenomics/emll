import pytensor.tensor
from emll.data_model_integration import create_noisy_observations_of_computed_values
from emll.data_model_integration import create_pytensor_from_data
import pytest 
import pytensor
import numpy as np
import pandas as pd
import pymc as pm

def test_create_noisy_observations_of_computed_values():
    """Tests the create_noisy_observations_of_computed_values function."""

    # setup test inputs
    n_rows = 10
    n_cols = 5
    input_string = "test"
    input_computed_tensor = pytensor.tensor.random.normal(size=(n_rows, n_cols))
    input_data = pd.DataFrame(np.random.rand(n_rows,n_cols))

    mismatched_input_data = pd.DataFrame(np.random.rand(n_rows+1,n_cols))
    input_stdev = pd.DataFrame(np.random.rand(n_rows,n_cols))
    mismatched_input_stdev_rows = pd.DataFrame(np.random.rand(n_rows+1,n_cols))
    mismatched_input_stdev_cols = pd.DataFrame(np.random.rand(n_rows,n_cols+1))
    negative_input_stdev = -1*pd.DataFrame(np.random.rand(n_rows,n_cols))

    # test that data and stdev shape mismatch raises a value error
    with pytest.raises(ValueError):
        create_noisy_observations_of_computed_values(input_string,input_computed_tensor,input_data,mismatched_input_stdev_rows)
    
    # test that data and tensor shape mismatch raises a value error
    with pytest.raises(ValueError):
        create_noisy_observations_of_computed_values(input_string,input_computed_tensor,mismatched_input_data,input_stdev)

    # test that negative stdev raises a value error
    with pytest.raises(ValueError):
        create_noisy_observations_of_computed_values(input_string,input_computed_tensor,input_data,negative_input_stdev)

    # test that data and stdev column mismatch raises a value error
    with pytest.raises(ValueError):
        create_noisy_observations_of_computed_values(input_string,input_computed_tensor,input_data,mismatched_input_stdev_cols)

    # test that data and stdev row mismatch raises a value error
    with pytest.raises(ValueError):
        create_noisy_observations_of_computed_values(input_string,input_computed_tensor,input_data,mismatched_input_stdev_rows)

    ## test function outputs

    # setup test inputs
    data_values = {
        'x': [1.0, np.inf, np.nan],
        'y': [np.inf, np.nan, 2]
    }

    stdev_values = {
        'x': [0.5, np.inf, np.nan],
        'y': [np.inf, np.nan, 2.5]
    }
    input_string = "test"
    input_data_df = pd.DataFrame(data_values)
    input_tensor = pytensor.tensor.random.normal(size=input_data_df.shape)
    input_tensor_values = input_tensor.eval()
    input_stdev = pd.DataFrame(stdev_values)


    # test that a value eror is raised if the function isn't run in a pymc model context
    with pytest.raises(ValueError):
        create_noisy_observations_of_computed_values(input_string, input_tensor, input_data_df, input_stdev)

    # check the outputs of the function
    test_model = pm.Model()
    with test_model:
        actual_output = create_noisy_observations_of_computed_values(input_string, input_tensor, input_data_df, input_stdev)

        # test that the actual and expected keys (row, column) are the same
        expected_output_keys = [(0,'x'),(2,'y')]
        actual_output_keys = list(actual_output)
        assert expected_output_keys == actual_output_keys

        # test that the actual and expected distribution name, mean and stdev are the same
        for key in expected_output_keys:
            row, col = key
            col_index = input_data_df.columns.get_loc(col)
            actual_distribution = actual_output[key]
            actual_name = actual_distribution.name
            actual_mean = actual_distribution.owner.inputs[3].eval()
            actual_stdev = actual_distribution.owner.inputs[4].eval()
            expected_name = f'{input_string}_{row}_{col}_obs'
            expected_mean = input_tensor_values[row,col_index]
            expected_stdev = input_stdev.iloc[row][col]
            assert actual_name == expected_name
            assert actual_mean == expected_mean
            assert actual_stdev == expected_stdev


def test_create_pytensor_from_data():

    # setup input name and data
    input_string = 'test'
    input_data_dict_mixed = {
        'x': [1.0, np.inf, np.nan],
        'y': [np.inf, np.nan, 2]
    }
    input_data_dict_all_observed = {
        'x': [1.0, 10.0, 100.0],
        'y': [-5, -25, -125]
    }
    input_dataframe_mixed = pd.DataFrame(input_data_dict_mixed)
    input_dataframe_all_observed = pd.DataFrame(input_data_dict_all_observed)

    # setup "wrong" inputs for error handling checks
    input_string_wrong_type= 42
    input_data_dict_mixed_wrong_type = {
        'x': [1.0, np.inf, None],
        'y': ["test", np.nan, 2]
    }
    input_dataframe_mixed_wrong_type = pd.DataFrame(input_data_dict_mixed_wrong_type)

    # check if type error is raised if name is not a string
    with pytest.raises(TypeError):
        create_pytensor_from_data(input_string_wrong_type,input_dataframe_mixed)
    
    # check if type error is raised if data is not a dataframe
    with pytest.raises(TypeError):
        create_pytensor_from_data(input_string,input_data_dict_mixed)

    # check that value error is raised if data values are not finite, np.inf, or np.nan
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_mixed_wrong_type)

    # check that type error is raised if not run in model context
    with pytest.raises(TypeError):
        create_pytensor_from_data(input_string,input_dataframe_mixed)


    test_model = pm.Model()
    with test_model:
        
        # check if actual returned value matches expected value 
        # case 1: all data is observed across all conditions
        # expect a normal distribution to be returned
            rv_tensor = create_pytensor_from_data(input_string,input_dataframe_all_observed)
        

    #     # check if returned value is a pytensor
    #     assert(type(create_pytensor_from_data(input_string,input_dataframe_mixed))==pytensor.tensor.TensorType)
    

    
    # check rows and columns

    # check if value error is raised if 
    #input_data_values = 1
    #raise NotImplementedError