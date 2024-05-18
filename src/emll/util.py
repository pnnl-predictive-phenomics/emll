import numpy as np
import scipy as sp
import pytensor.tensor as at
import pymc as pm

from pytensor.graph.basic import ancestors
from pytensor.tensor.variable import TensorVariable
from pytensor.tensor.random.op import RandomVariable 


def create_elasticity_matrix(model):
    """Create an elasticity matrix given the model in model.

    E[j,i] represents the elasticity of reaction j for metabolite i.

    """

    n_metabolites = len(model.metabolites)
    n_reactions = len(model.reactions)
    array = np.zeros((n_reactions, n_metabolites), dtype=float)

    m_ind = model.metabolites.index
    r_ind = model.reactions.index

    for reaction in model.reactions:
        for metabolite, stoich in reaction.metabolites.items():
            # Reversible reaction, assign all elements to -stoich
            if reaction.reversibility:
                array[r_ind(reaction), m_ind(metabolite)] = -np.sign(stoich)

            # Irrevesible in forward direction, only assign if met is reactant
            elif (not reaction.reversibility) & (reaction.upper_bound > 0) & (stoich < 0):
                array[r_ind(reaction), m_ind(metabolite)] = -np.sign(stoich)

            # Irreversible in reverse direction, only assign in met is product
            elif (not reaction.reversibility) & (reaction.lower_bound < 0) & (stoich > 0):
                array[r_ind(reaction), m_ind(metabolite)] = -np.sign(stoich)

    return array


def create_Ey_matrix(model):
    """This function should return a good guess for the Ey matrix. This
    essentially requires considering the effects of the reactants / products
    for the unbalanced exchange reactions, and is probably best handled
    manually for now."""

    boundary_indexes = [model.reactions.index(r) for r in model.medium.keys()]
    boundary_directions = [
        1 if r.products else -1 for r in model.reactions.query(lambda x: x.boundary, None)
    ]
    ny = len(boundary_indexes)
    Ey = np.zeros((len(model.reactions), ny))

    for i, (rid, direction) in enumerate(zip(boundary_indexes, boundary_directions)):
        Ey[rid, i] = direction

    return Ey


def compute_waldherr_reduction(N, tol=1e-8):
    """Uses the SVD to calculate a reduced stoichiometric matrix, link, and
    conservation matrices.

    Returns:
    Nr, L, G

    """
    u, e, vh = sp.linalg.svd(N)

    E = sp.linalg.diagsvd(e, *N.shape)

    if len(e) is not E.shape[0]:
        e_new = np.zeros(N.shape[0])
        e_new[: len(e)] = e
        e = e_new

    Nr = E[e > tol] @ vh
    L = u[:, e > tol]
    G = u[:, e >= tol]

    return Nr, L, G


def compute_smallbone_reduction(N, Ex, v_star, tol=1e-8):
    """Uses the SVD to calculate a reduced stoichiometric matrix, then
    calculates a link matrix as described in Smallbone *et al* 2007.

    Returns:
    Nr, L, P

    """
    q, r, p = sp.linalg.qr((N @ np.diag(v_star) @ Ex).T, pivoting=True)

    # Construct permutation matrix
    P = np.zeros((len(p), len(p)), dtype=int)
    for i, pi in enumerate(p):
        P[i, pi] = 1

    # Get the matrix rank from the r matrix
    maxabs = np.max(np.abs(np.diag(r)))
    maxdim = max(N.shape)
    tol = maxabs * maxdim * (2.220446049250313e-16)

    # Find where the rows of r are all less than tol
    rank = (~(np.abs(r) < tol).all(1)).sum()

    Nr = P[:rank] @ N
    L = N @ np.linalg.pinv(Nr)

    return Nr, L, P


def initialize_elasticity(
    N, name=None, b=0.01, alpha=5, sigma=1, m_compartments=None, r_compartments=None
):
    """Initialize the elasticity matrix, adjusting priors to account for
    reaction stoichiometry. Uses `SkewNormal(mu=0, sd=sd, alpha=sign*alpha)`
    for reactions in which a metabolite participates, and a `Laplace(mu=0,
    b=b)` for off-target regulation.

    Also accepts compartments for metabolites and reactions. If given,
    metabolites are only given regulatory priors if they come from the same
    compartment as the reaction.

    Parameters
    ==========

    N : np.ndarray
        A (nm x nr) stoichiometric matrix for the given reactions and metabolites
    name : string
        A name to be used for the returned pymc3 probabilities
    b : float
        Hyperprior to use for the Laplace distributions on regulatory interactions
    alpha : float
        Hyperprior to use for the SkewNormal distributions. As alpha ->
        infinity, these priors begin to resemble half-normal distributions.
    sigma : float
        Scale parameter for the SkewNormal distribution.
    m_compartments : list
        Compartments of metabolites. If None, use a densely connected
        regulatory prior.
    r_compartments : list
        Compartments of reactions

    Returns
    =======

    E : pymc3 matrix
        constructed elasticity matrix

    """

    if name is None:
        name = "ex"

    if m_compartments is not None:
        assert r_compartments is not None, "reaction and metabolite compartments must both be given"

        regulation_array = np.array(
            [[a in b for a in m_compartments] for b in r_compartments]
        ).flatten()

    else:
        # If compartment information is not given, assume all metabolites and
        # reactions are in the same compartment
        regulation_array = np.array([True] * (N.shape[0] * N.shape[1]))

    # Guess an elasticity matrix from the smallbone approximation
    e_guess = -N.T

    # Find where the guessed E matrix has zero entries
    e_flat = e_guess.flatten()
    nonzero_inds = np.where(e_flat != 0)[0]
    offtarget_inds = np.where(e_flat == 0)[0]
    e_sign = np.sign(e_flat[nonzero_inds])

    # For the zero entries, determine whether regulation is feasible based on
    # the compartment comparison
    offtarget_reg = regulation_array[offtarget_inds]
    reg_inds = offtarget_inds[offtarget_reg]
    zero_inds = offtarget_inds[~offtarget_reg]

    num_nonzero = len(nonzero_inds)
    num_regulations = len(reg_inds)
    num_zeros = len(zero_inds)

    # Get an index vector that 'unrolls' a stacked [kinetic, capacity, zero]
    # vector into the correct order
    flat_indexer = np.hstack([nonzero_inds, reg_inds, zero_inds]).argsort()

    if alpha is not None:
        e_kin_entries = pm.SkewNormal(
            name + "_kinetic_entries",
            sigma=sigma,
            alpha=alpha,
            shape=num_nonzero,
            initval=0.1 + np.abs(np.random.randn(num_nonzero)),
        )
    else:
        e_kin_entries = pm.HalfNormal(
            name + "_kinetic_entries",
            sigma=sigma,
            shape=num_nonzero,
            initval=0.1 + np.abs(np.random.randn(num_nonzero)),
        )

    e_cap_entries = pm.Laplace(
        name + "_capacity_entries",
        mu=0,
        b=b,
        shape=num_regulations,
        initval=b * np.random.randn(num_regulations),
    )

    flat_e_entries = at.concatenate(
        [
            e_kin_entries * e_sign,  # kinetic entries
            e_cap_entries,  # capacity entries
            at.zeros(num_zeros),
        ]
    )  # different compartments

    E = flat_e_entries[flat_indexer].reshape(N.T.shape)

    return E



def assert_tensor_equal(expected_tensor, actual_tensor, check_name=True):
    """
    Check if two tensors are equal in shape, random variables, and random variable parameters.

    Parameters:
        expected_tensor (TensorVariable): The expected tensor.
        actual_tensor (TensorVariable): The actual tensor.
        check_name (bool, optional): Whether to check if the names of random variables are equal. Defaults to True.

    Raises:
        ValueError: If the shape of the expected tensor is not equal to the shape of the actual tensor.
        ValueError: If the number of random variables in the expected tensor is not equal to the number of random variables in the actual tensor.
        ValueError: If the names of the random variables in the expected tensor are not equal to the names of the random variables in the actual tensor.

    """

    # check that two tensors have the same shape, RVs, and RV parameters
    # optional: check names are equal

    if (expected_tensor.shape.eval() != actual_tensor.shape.eval()).all():
        raise ValueError(f"expected tensor shape {expected_tensor.shape.eval()} != actual tensor shape {actual_tensor.shape.eval()}")


    # Traverse the computational graph to get only the random variables
    expected_ancestor_nodes = ancestors([expected_tensor])
    expected_apply_nodes = [node for node in expected_ancestor_nodes if isinstance(node, TensorVariable) and node.owner is not None]
    expected_rv_nodes = [node for node in expected_apply_nodes if isinstance(node.owner.op, RandomVariable)]

    actual_ancestor_nodes = ancestors([actual_tensor])
    actual_apply_nodes = [node for node in actual_ancestor_nodes if isinstance(node, TensorVariable) and node.owner is not None]
    actual_rv_nodes = [node for node in actual_apply_nodes if isinstance(node.owner.op, RandomVariable)]

    expected_rv_shapes = [rv.shape.eval() for rv in expected_rv_nodes]
    actual_rv_shapes = [rv.shape.eval() for rv in actual_rv_nodes]
    print(f"expected: {expected_rv_shapes}")
    print(f"actual: {actual_rv_shapes}")

    # check the expected and actual number of RVs is equal
    if len(expected_rv_nodes) != len(actual_rv_nodes):
        raise ValueError(f"number of expected rvs { len(expected_rv_nodes)} != number of actual rvs {len(actual_rv_nodes)}") 

    actual_rv_names = {node.name for node in actual_rv_nodes}
    expected_rv_names = {node.name for node in expected_rv_nodes}

    # check names of random variables
    if len(actual_rv_names) != len(expected_rv_names):
        raise ValueError(f"length of expected names {len(expected_rv_names)} != actual {len(actual_rv_names)}")

    if check_name and actual_rv_names!=expected_rv_names:
        raise ValueError(f"{expected_rv_names-actual_rv_names} in expected but not actual. {actual_rv_names-expected_rv_names} in actual but not expected.")



    # check same number Normal

    # # Initialize counters for the number of each type of RV
    # num_normal = num_laplace = num_zeros = 0

    # # Initialize a separate index to track the position in the dataframe
    # dataframe_idx = 0

    # for idx, rv_node in enumerate(rv_nodes):
    #     # Find the next non-NaN entry in the dataframe
    #     while pd.isna(input_dataframe_mixed.values.flat[dataframe_idx]):
    #         dataframe_idx += 1  # Skip NaN entries (excluded variables)

    #     # Get the row and column names from the non-NaN position
    #     row_idx, col_idx = np.unravel_index(dataframe_idx, input_dataframe_mixed.shape)
    #     row_name = input_dataframe_mixed.index[row_idx]
    #     col_name = input_dataframe_mixed.columns[col_idx]

    #     # Get the expected name for the RV
    #     expected_name = f"{input_string}_{row_name}_{col_name}"

    #     # get the data value, stdev, and laplace params
    #     value = input_dataframe_mixed.iloc[row_idx, col_idx]
    #     stdev = input_stdev_dataframe_mixed.iloc[row_idx, col_idx]
    #     laplace_params = input_laplace_dataframe_mixed.iloc[row_idx, col_idx]

    #     # check the actual random variable type (normal), name, mean, and stdev matches expected
    #     if np.isfinite(value):  # Observed data (Normal RV)
    #         num_normal += 1
    #         expected_mu = value
    #         expected_sigma = stdev
    #         assert rv_node.owner.op.name == 'normal', f"RV is not a normal distribution: {rv_node.owner.op.name}"
    #         assert rv_node.name == expected_name, f"RV name mismatch: expected {expected_name}, got {rv_node.name}"
    #         assert np.isclose(rv_node.owner.inputs[3].eval(), expected_mu), f"RV mu mismatch: expected {expected_mu}, got {rv_node.owner.inputs[3].eval()}"
    #         assert np.isclose(rv_node.owner.inputs[4].eval(), expected_sigma), f"RV sigma mismatch: expected {expected_sigma}, got {rv_node.owner.inputs[4].eval()}"

    #     # check the actual random variable type (Laplace), name, loc, and scale matches expected
    #     elif np.isinf(value):  # Unobserved data (Laplace RV)
    #         num_laplace += 1
    #         expected_loc, expected_scale = laplace_params
    #         assert rv_node.owner.op.name == 'laplace', f"RV is not a Laplace distribution: {rv_node.owner.op.name}"
    #         assert rv_node.name == expected_name, f"RV name mismatch: expected {expected_name}, got {rv_node.name}"
    #         assert np.isclose(rv_node.owner.inputs[3].eval(), expected_loc), f"RV loc mismatch: expected {expected_loc}, got {rv_node.owner.inputs[3].eval()}"
    #         assert np.isclose(rv_node.owner.inputs[4].eval(), expected_scale), f"RV scale mismatch: expected {expected_scale}, got {rv_node.owner.inputs[4].eval()}"

    #     elif pd.isna(value):  # Excluded data (Zeros)
    #         num_zeros += 1

    #     # Increment the dataframe index for the next iteration
    #     dataframe_idx += 1

    # # Evaluate the tensor and count the zeros
    # tensor_values = data_tensor.eval()
    # actual_num_zeros = np.sum(tensor_values == 0)

    # # Check the counts of normal, Laplace, and zero values match expected
    # expected_num_normal = np.isfinite(input_dataframe_mixed.values).sum()
    # expected_num_laplace = np.isinf(input_dataframe_mixed.values).sum()
    # expected_num_zeros = pd.isna(input_dataframe_mixed.values).sum()
    # assert num_normal == expected_num_normal, f"Expected {expected_num_normal} normal RVs, found {num_normal}"
    # assert num_laplace == expected_num_laplace, f"Expected {expected_num_laplace} Laplace RVs, found {num_laplace}"
    # assert actual_num_zeros == expected_num_zeros, f"Expected {expected_num_zeros} zeros, found {actual_num_zeros}"

    # pass