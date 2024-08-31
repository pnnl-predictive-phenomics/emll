import numpy as np
import scipy as sp
import pytensor.tensor as at
import pymc as pm
from pathlib import Path

import tellurium as te
import libsbml

import os


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
            elif (
                (not reaction.reversibility) & (reaction.upper_bound > 0) & (stoich < 0)
            ):
                array[r_ind(reaction), m_ind(metabolite)] = -np.sign(stoich)

            # Irreversible in reverse direction, only assign in met is product
            elif (
                (not reaction.reversibility) & (reaction.lower_bound < 0) & (stoich > 0)
            ):
                array[r_ind(reaction), m_ind(metabolite)] = -np.sign(stoich)

    return array


def create_Ey_matrix(model):
    """This function should return a good guess for the Ey matrix. This
    essentially requires considering the effects of the reactants / products
    for the unbalanced exchange reactions, and is probably best handled
    manually for now."""

    boundary_indexes = [model.reactions.index(r) for r in model.medium.keys()]
    boundary_directions = [
        1 if r.products else -1
        for r in model.reactions.query(lambda x: x.boundary, None)
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
        assert (
            r_compartments is not None
        ), "reaction and metabolite compartments must both be given"

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


def ant_to_cobra(antimony_path):
    """
    This method takes in an Antimony file and converts it to a Cobra-
    friendly format by removing all boundary species and replacing all
    rate laws with a constant. Both an Antimony file and an SBML are
    produced, and the file name is returned.
    """
    output_name = antimony_path.split("/")[-1]
    output_name = output_name.split(".")[0]
    output_path = Path(antimony_path).parent

    # load normal Antimony file
    with open(antimony_path, "r") as file:
        # reprint the antimony model into a temp file to ensure proper
        # headers
        # load antimony file
        r = te.loada(antimony_path)
        # print antimony file to temp
        with open("tempA7K8L2P4W9.txt", "w") as f:
            f.write(r.getCurrentAntimony())
        old_ant = "tempA7K8L2P4W9.txt"

    with open(old_ant, "r") as file:
        lines = file.readlines()
        section_indices = []
        c_index = -1
        for i, line in enumerate(lines):
            if "//" in line:
                section_indices.append(i)
            if "// Compartment" in line and c_index == -1:
                c_index = i
        next_section = section_indices.index(c_index) + 1

        with open(old_ant, "r") as file:
            lines = file.readlines()[c_index : section_indices[next_section]]

        with open(f"{output_path}/{output_name}_cobra.ant", "w") as f:
            f.write("")

        for line in lines:
            line = line.strip()
            if "$" not in line:
                with open(f"{output_path}/{output_name}_cobra.ant", "a") as f:
                    f.write(line + "\n")
            else:
                no_bd_sp = [i for i in line.split(",") if "$" not in i]
                no_bd_sp = [i for i in no_bd_sp if i != ""]

                line = ",".join(no_bd_sp)
                if line != "" and line[-1] != ";":
                    line += ";"
                if "species" not in line:
                    line = "species" + line
                with open(f"{output_path}/{output_name}_cobra.ant", "a") as f:
                    f.write(line + "\n")

    r = te.loada(antimony_path)
    doc = libsbml.readSBMLFromString(r.getSBML())
    model = doc.getModel()

    bd_sp = r.getBoundarySpeciesIds()

    reactants_list = []
    products_list = []

    for n in range(len(r.getReactionIds())):
        rxn = model.getReaction(n)
        reactants = []
        products = []

        for reactant in range(rxn.getNumReactants()):
            stoich = rxn.getReactant(reactant).getStoichiometry()
            if stoich == 1:
                reactants.append(rxn.getReactant(reactant).species)
            else:
                reactants.append(str(stoich) + " " + rxn.getReactant(reactant).species)
        reactants_list.append([i for i in reactants if i not in bd_sp])

        for product in range(rxn.getNumProducts()):
            stoich = rxn.getProduct(product).getStoichiometry()
            if stoich == 1:
                products.append(rxn.getProduct(product).species)
            else:
                products.append(str(stoich) + " " + rxn.getProduct(product).species)
        products_list.append([i for i in products if i not in bd_sp])

    for i in range(len(reactants_list)):
        r1 = " + ".join(reactants_list[i])
        p1 = " + ".join(products_list[i])
        with open(f"{output_path}/{output_name}_cobra.ant", "a") as f:
            f.write(r.getReactionIds()[i] + ": " + r1 + " -> " + p1 + "; 1;\n")

    with open(f"{output_path}/{output_name}_cobra.ant", "a") as f:
        f.write("\n")

    for sp in r.getFloatingSpeciesIds():
        with open(f"{output_path}/{output_name}_cobra.ant", "a") as f:
            f.write(sp + " = 1;\n")

    with open(f"{output_path}/{output_name}_cobra.xml", "w") as f:
        f.write(te.loada(f"{output_path}/{output_name}_cobra.ant").getCurrentSBML())

    if "tempA7K8L2P4W9.txt" in os.listdir():
        os.remove("tempA7K8L2P4W9.txt")

    return f"{output_path}/{output_name}_cobra"
