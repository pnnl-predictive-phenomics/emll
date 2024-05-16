
from emll.data_model_integration import create_noisy_observations_of_computed_values
from emll.data_model_integration import create_pytensor_from_data
import pytest 
import pytensor
import numpy as np
import pandas as pd
import pymc as pm

from pytensor.graph.basic import ancestors
from pytensor.tensor.variable import TensorVariable
from pytensor.tensor.random.op import RandomVariable 



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
    """Tests the create_pytensor_from_data function."""
    ### setup initial fixtures ###
    input_string = 'test'
    input_data_dict_all_observed = {
        'x': [1.0, 10.0, 100.0],
        'y': [-5, -25, -125]
    }
    input_dataframe_all_observed = pd.DataFrame(input_data_dict_all_observed)
    input_stdev_dict_all_observed = {
        'x': [0.25, 2.5, 25],
        'y': [1, 2, 4]
    }
    input_stdev_dataframe_all_observed = pd.DataFrame(input_stdev_dict_all_observed)

    input_laplace_dict_no_observed = {
        'x': [(0, 1), (10, 0.5), (100, 0.25)],
        'y': [(-0.5, 4), (-0.25, 2), (-0.125, 1)]
    }
    input_laplace_dataframe_no_observed = pd.DataFrame(input_laplace_dict_no_observed)

    ### check that errors are caught from incorrect inputs ###
    # check if type error is raised if name is not a string
    input_string_wrong_type = 42
    with pytest.raises(TypeError):
        create_pytensor_from_data(input_string_wrong_type,input_dataframe_all_observed,input_stdev_dataframe_all_observed, input_laplace_dataframe_no_observed)
    
    # check if type error is raised if data is not a dataframe
    with pytest.raises(TypeError):
        create_pytensor_from_data(input_string,input_data_dict_all_observed,input_stdev_dataframe_all_observed, input_laplace_dataframe_no_observed)

    # check if type error is raised if the stdev is not a dataframe
    with pytest.raises(TypeError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dict_all_observed, input_laplace_dataframe_no_observed)

    # check if type error is raised if the laplace is not a dataframe
    with pytest.raises(TypeError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_all_observed, input_laplace_dict_no_observed)

    # check if value error is raised if the shape of the data and stdev dataframes aren't equal
    input_stdev_dataframe_fewer_columns = input_stdev_dataframe_all_observed.drop(columns='x').copy(deep=True)
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_fewer_columns, input_laplace_dataframe_no_observed)

    # check if value error is raised if the shape of the data and laplace dataframes aren't equal
    input_laplace_dataframe_fewer_columns = input_laplace_dataframe_no_observed.drop(columns='x').copy(deep=True)
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_all_observed, input_laplace_dataframe_fewer_columns)

    # check if value error is raised if data values are not finite, np.inf, or np.nan
    input_data_dict_wrong_values = {
        'x': [1.0, np.inf, None],
        'y': ["test", np.nan, -2]
    }
    input_dataframe_wrong_values = pd.DataFrame(input_data_dict_wrong_values)
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_wrong_values,input_stdev_dataframe_all_observed, input_laplace_dataframe_no_observed)

    # check that a value error is raised if stdev values are not None or finite > 0
    input_stdev_dataframe_wrong_values = input_dataframe_wrong_values.copy(deep=True)
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_wrong_values, input_laplace_dataframe_no_observed)

    # check if value error is raised if the laplace dataframe values are not a (finite, postive finite) or np.nan
    input_laplace_dict_wrong_values = {
        'x': [(0, 1), (10, 0.5, 2.5), "test"],
        'y': [np.inf, (-0.25, 2), np.nan]
    }
    input_laplace_dataframe_wrong_values = pd.DataFrame(input_laplace_dict_wrong_values)
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_all_observed, input_laplace_dataframe_wrong_values)

    # check that a value error is raised if the stdev columns don't match the data
    input_stdev_dataframe_wrong_cols = input_stdev_dataframe_all_observed.copy(deep=True).rename(columns={'x': 'z'})
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_wrong_cols, input_laplace_dataframe_no_observed)

    # check that a value error is raised if the stdev rows don't match the data
    input_stdev_dataframe_renamed_index = input_stdev_dataframe_all_observed.copy(deep=True).rename(index={i: f'row{i+1}' for i in range(len(input_stdev_dataframe_all_observed))})
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_renamed_index, input_laplace_dataframe_no_observed)

    # check that a value error is raised if the laplace columns don't match the data
    input_laplace_dataframe_wrong_cols = input_laplace_dataframe_no_observed.copy(deep=True).rename(columns={'x': 'z'})
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_all_observed, input_laplace_dataframe_wrong_cols)

    # check that a value error is raised if the laplace rows don't match the data
    input_laplace_dataframe_renamed_index = input_laplace_dataframe_no_observed.copy(deep=True).rename(index={i: f'row{i+1}' for i in range(len(input_laplace_dataframe_no_observed))})
    with pytest.raises(ValueError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_all_observed, input_laplace_dataframe_renamed_index)
    
    # check that type error is raised if not run in model context
    with pytest.raises(TypeError):
        create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_all_observed, input_laplace_dataframe_no_observed)


    ### check actual outputs match expected outputs ###

    # Test case 1: all model variables are observed for all conditions
    # Laplace values aren't used if all variables are observed
    input_laplace_dict_all_observed = {
        'x': [np.nan, np.nan, np.nan],
        'y': [np.nan, np.nan, np.nan]
    }
    input_laplace_dataframe_all_observed = pd.DataFrame(input_laplace_dict_all_observed)
    test_model = pm.Model()
    with test_model:
        data_tensor = create_pytensor_from_data(input_string,input_dataframe_all_observed,input_stdev_dataframe_all_observed, input_laplace_dataframe_all_observed)
        
        # traverse the computational graph to get only the random variables
        ancestor_nodes = ancestors([data_tensor])
        apply_nodes = [node for node in ancestor_nodes if isinstance(node, TensorVariable) and node.owner is not None]
        rv_nodes = [node for node in apply_nodes if isinstance(node.owner.op, RandomVariable)]

        # Check the total number of RVs (1 per data point)
        expected_rv_count = np.prod(input_dataframe_all_observed.shape)
        assert len(rv_nodes) == expected_rv_count, f"Expected {expected_rv_count} RVs, found {len(rv_nodes)}"

        # Check the RV type, name, mu, and sigma
        for idx, rv_node in enumerate(rv_nodes):
            # Get the expected row name (index) and column name
            row_idx, col_idx = np.unravel_index(idx, input_dataframe_all_observed.shape)
            row_name = input_dataframe_all_observed.index[row_idx]
            col_name = input_dataframe_all_observed.columns[col_idx]

            # get the expected name, mean (mu), and stdev (sigma)
            expected_rv_type = 'normal'
            expected_name = f"{input_string}_{row_name}_{col_name}"
            expected_mu = input_dataframe_all_observed.iloc[row_idx, col_idx]
            expected_sigma = input_stdev_dataframe_all_observed.iloc[row_idx, col_idx]

            # check random variable distribute type, name, mean, and stdev
            assert rv_node.owner.op.name == expected_rv_type, f"RV is not a normal distribution: {rv_node.owner.op.name}"
            assert rv_node.name == expected_name, f"RV name mismatch: expected {expected_name}, got {rv_node.name}"
            assert np.isclose(rv_node.owner.inputs[3].eval(), expected_mu), f"RV mu mismatch: expected {expected_mu}, got {rv_node.owner.inputs[3].eval()}"
            assert np.isclose(rv_node.owner.inputs[4].eval(), expected_sigma), f"RV sigma mismatch: expected {expected_sigma}, got {rv_node.owner.inputs[4].eval()}"

    # Test case 2: all model variables are unobserved for all conditions
    # unobserved data should be np.inf, and normal distrubition / stdev isn't used
    input_dict_no_observed = {
        'x': [np.inf, np.inf, np.inf],
        'y': [np.inf, np.inf, np.inf]
    }
    input_dataframe_no_observed = pd.DataFrame(input_dict_no_observed)
    input_stdev_dict_no_observed = {
        'x': [np.nan, np.nan, np.nan],
        'y': [np.nan, np.nan, np.nan]
    }
    input_stdev_dataframe_no_observed = pd.DataFrame(input_stdev_dict_no_observed)

    test_model = pm.Model()
    with test_model:
        data_tensor = create_pytensor_from_data(input_string,input_dataframe_no_observed,input_stdev_dataframe_no_observed, input_laplace_dataframe_no_observed)
        print(pytensor.dprint(data_tensor))
        # traverse the computational graph to get only the random variables
        ancestor_nodes = ancestors([data_tensor])
        apply_nodes = [node for node in ancestor_nodes if isinstance(node, TensorVariable) and node.owner is not None]
        rv_nodes = [node for node in apply_nodes if isinstance(node.owner.op, RandomVariable)]

        # Check the total number of RVs (1 per data point)
        expected_rv_count = np.prod(input_dataframe_no_observed.shape)
        assert len(rv_nodes) == expected_rv_count, f"Expected {expected_rv_count} RVs, found {len(rv_nodes)}"

        # Check the RV type, name, mu, and sigma
        for idx, rv_node in enumerate(rv_nodes):
            # Get the expected row name (index) and column name
            row_idx, col_idx = np.unravel_index(idx, input_dataframe_no_observed.shape)
            row_name = input_dataframe_no_observed.index[row_idx]
            col_name = input_dataframe_no_observed.columns[col_idx]

            # get the expected name, location (laplace mu), and scale (laplace b)
            expected_rv_type = 'laplace'
            expected_name = f"{input_string}_{row_name}_{col_name}"
            expected_loc = input_laplace_dataframe_no_observed.iloc[row_idx, col_idx][0]
            expected_scale = input_laplace_dataframe_no_observed.iloc[row_idx, col_idx][1]

            # check random variable distribute type, name, location, and scale
            assert rv_node.owner.op.name == expected_rv_type, f"RV is not a Laplace distribution: {rv_node.owner.op.name}"
            assert rv_node.name == expected_name, f"RV name mismatch: expected {expected_name}, got {rv_node.name}"
            assert np.isclose(rv_node.owner.inputs[3].eval(), expected_loc), f"RV loc mismatch: expected {expected_mu}, got {rv_node.owner.inputs[3].eval()}"
            assert np.isclose(rv_node.owner.inputs[4].eval(), expected_scale), f"RV scale mismatch: expected {expected_sigma}, got {rv_node.owner.inputs[4].eval()}"
        

    # Test case 3: all model variables are excluded for all conditions
    # All data should be np.nan with Normal and Laplace distributions not used
    input_data_dict_all_excluded = {
        'x': [np.nan, np.nan, np.nan],
        'y': [np.nan, np.nan, np.nan]
    }
    input_dataframe_all_excluded = pd.DataFrame(input_data_dict_all_excluded)

    test_model = pm.Model()
    with test_model:
        data_tensor = create_pytensor_from_data(input_string, input_dataframe_all_excluded, input_stdev_dataframe_no_observed, input_laplace_dataframe_no_observed)

        # check the actual number of zeros is the same as expected
        tensor_values = data_tensor.eval()
        actual_num_zeros = np.sum(tensor_values == 0)
        expected_num_zeros = np.prod(input_dataframe_all_excluded.shape)
        assert actual_num_zeros == expected_num_zeros, f"Expected {expected_num_zeros} zeroes, found {actual_num_zeros}"

        

        # # test case 4: variables are a mixture of observed, unobserved, and excluded for all conditions
    # input_data_dict_mixed = {
    #     'x': [1.0, np.inf, np.nan],
    #     'y': [np.inf, np.nan, 2]
    # }
    # input_dataframe_mixed = pd.DataFrame(input_data_dict_mixed)
    # input_stdev_dict_mixed = {
    #     'x': [0.25, np.nan, np.nan],
    #     'y': [np.nan, np.nan, 4]
    # }
    # input_stdev_dataframe_mixed = pd.DataFrame(input_stdev_dict_mixed)
