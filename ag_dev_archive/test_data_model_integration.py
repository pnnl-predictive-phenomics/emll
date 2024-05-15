import pytensor.tensor as tensor 
from pytensor.tensor.random.utils import RandomStream
from pytensor import function

from emll.data_model_integration import create_noisy_observations_of_computed_values
import pytest 
import numpy as np
import pandas as pd
import pymc as pm

import pytensor


import cobra

# def compare_tensors(expected_tensor:pytensor.tensor, actual_tensor:pytensor.tensor):
#     """
#     Compare two PyMC tensors to ensure they have the same type, shape,
#     and distributions with identical parameters.

#     Parameters:
#     expected_tensor : PyTensor tensor
#         The tensor representing the expected values and distributions.
#     actual_tensor : PyTensor tensor
#         The tensor representing the actual values and distributions to compare.

#     Raises:
#     AssertionError
#         If any of the checks fail.
#     """
#     # TODO: review this code!
#     def get_random_variables(tensor):
#         """Traverse the computational graph of a tensor to extract RandomVariable nodes."""
#         random_vars = []

#         def traverse(node):
#             if node is None:
#                 return
#             if isinstance(node.op, RandomVariable):
#                 random_vars.append(node)
#             for input in node.inputs:
#                 traverse(input.owner)

#         if tensor.owner is not None:
#             traverse(tensor.owner)
#         return random_vars


#     def compare_random_variables(rv1, rv2):
#         """Compare two lists of random variables to check if they have the same distributions and parameters."""
#         if len(rv1) != len(rv2):
#             return False, "Different number of random variables"

#         for rvar1, rvar2 in zip(rv1, rv2):
#             # Check if the distribution types are the same
#             if type(rvar1.op) != type(rvar2.op):
#                 return False, f"Different distribution types: {type(rvar1.op)} vs {type(rvar2.op)}"

#             # Check if the random generators are the same
#             if not np.array_equal(rvar1.inputs[1].eval(), rvar2.inputs[1].eval()):
#                 return False, "Different random generators"
            
#             # Check if the size is the same
#             if not rvar1.inputs[1].equals(rvar2.inputs[1]):
#                 return False, "Different sizes"
            
#             # Check if the dtype is the same
#             if not rvar1.inputs[2].equals(rvar2.inputs[2]):
#                 return False, "Different dtypes"

#             # Compare the distribution parameters
#             for param1, param2 in zip(rvar1.inputs[3:], rvar2.inputs[3:]):  # Skip the first three inputs (rng, size, dtype)
#                 if not param1.equals(param2):
#                     return False, f"Different distribution parameters: {param1} vs {param2}"
                
#             # TODO: Evaluate RVs and get log-p

#         return True, "Random variables and their parameters match"

#     # Use the function to compare the random variables from both tensors
#     rv_comparison_result, rv_comparison_message = compare_random_variables(
#         get_random_variables(expected_tensor),
#         get_random_variables(actual_tensor)
#     )

#     assert rv_comparison_result, rv_comparison_message

#     # Compare the types of the tensors
#     assert type(expected_tensor) == type(actual_tensor), "Tensor types do not match."

#     # Compare the shapes of the tensors
#     assert np.array_equal(expected_tensor.shape.eval(), actual_tensor.shape.eval()), "Tensor shapes do not match."

#     return True


# #@pytest.fixture
# def expected_results_from_hackett():
#     # Load model and data
#     seed = 1
#     rng = np.random.default_rng(seed)
#     np.random.seed(seed)


#     model = cobra.io.load_yaml_model("./notebooks/data/jol2012.yaml")
#     rxn_compartments = [r.compartments if "e" not in r.compartments else "t" for r in model.reactions]
#     rxn_compartments[model.reactions.index("SUCCt2r")] = "c"
#     rxn_compartments[model.reactions.index("ACt2r")] = "c"
#     for rxn in model.exchanges:
#         rxn_compartments[model.reactions.index(rxn)] = "t"

#     fluxes = pd.read_csv("./notebooks/data/boundary_fluxes.csv", index_col=0)
#     enzymes = pd.read_csv("./notebooks/data/enzyme_measurements.csv", index_col=0)
#     to_consider = fluxes.columns
#     enzymes = enzymes.loc[:, to_consider]
#     n_exp = len(to_consider) - 1
#     ref_state = "P0.11"

#     enzymes_norm = (2 ** enzymes.subtract(enzymes["P0.11"], 0)).T
#     enzymes_norm = enzymes_norm.drop(ref_state)

#     enzymes_inds = np.array([model.reactions.index(rxn) for rxn in enzymes_norm.columns])
#     enzymes_laplace_inds = []
#     enzymes_zero_inds = []
#     for i, rxn in enumerate(model.reactions):
#         if rxn.id not in enzymes_norm.columns:
#             if ("e" not in rxn.compartments) and (len(rxn.compartments) == 1):
#                 enzymes_laplace_inds += [i]
#             else:
#                 enzymes_zero_inds += [i]
#     enzymes_laplace_inds = np.array(enzymes_laplace_inds)
#     enzymes_zero_inds = np.array(enzymes_zero_inds)
#     enzymes_indexer = np.hstack([enzymes_inds, enzymes_laplace_inds, enzymes_zero_inds]).argsort()


#     with pm.Model() as pymc_model:
#         enzymes_measured = pm.Normal("log_e_measured", mu=np.log(enzymes_norm), sigma=0.2, shape=(n_exp, len(enzymes_inds)), rng=rng)
#         enzymes_unmeasured = pm.Laplace("log_e_unmeasured", mu=0, b=0.1, shape=(n_exp, len(enzymes_laplace_inds)), rng=rng)
#         log_enzymes_norm_tensor = T.concatenate(
#             [enzymes_measured, enzymes_unmeasured, T.zeros((n_exp, len(enzymes_zero_inds)))], axis=1
#         )[:, enzymes_indexer]
    
#     model_params = {
#         "measured_name":"log_e_measured", 
#         "measured_mu":np.log(enzymes_norm),
#         "measured_sigma":0.2,
#         "measured_shape":(n_exp, len(enzymes_inds)),
#         "unmeasured_name":"log_e_unmeasured",
#         "unmeasured_mu":0,
#         "unmeasured_b":0.1,
#         "unmeasured_shape":(n_exp, len(enzymes_laplace_inds)),
#         "excluded_shape":(n_exp, len(enzymes_zero_inds)), 
#         "tensor_indexer":enzymes_indexer,
#     }    
#     return (log_enzymes_norm_tensor, model_params)


# def test_create_pytensor_from_data(expected_results_from_hackett):
#     '''Tests if pytensor generated from data matches expectations.'''
#     expected_tensor = expected_results_from_hackett[0]
#     hacket_model_parameters = expected_results_from_hackett[1]
#     actual_tensor = data_model_integration.create_pytensor_from_data(hacket_model_parameters)
#     assert compare_tensors(expected_tensor,actual_tensor) == True


def test_create_noisy_observations_of_computed_values():
    """Tests the create_noisy_observations_of_computed_values function."""

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



