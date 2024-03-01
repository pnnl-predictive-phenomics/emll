import numpy as np
import pymc as pm
from . import util, linlog_model
from cobra.util import create_stoichiometric_matrix


def build_unconditional_pymc_model(cobra_model):
    """Build unconditional pymc model from cobra model only. This relies on elasticity matrices
    and stoichiometric matrices only, and is helpful for sampling prior predictives."""

    # Get full elasticity matrix
    Ex = util.create_elasticity_matrix(cobra_model)

    # Get external elasticity matrix
    Ey = util.create_Ey_matrix(cobra_model)

    # Get stoichiometric matrix
    N = create_stoichiometric_matrix(cobra_model, dtype='int')

    with pm.Model() as pymc_model:
        # Initialize elasticities
        Ex_t = pm.Deterministic('Ex', util.initialize_elasticity(N, 'ex', b=0.05, sigma=1, alpha=5))
        Ey_t = pm.Deterministic('Ey', util.initialize_elasticity(-Ey.T, 'ey', b=0.05, sigma=1, alpha=5))

    return pymc_model


def sample_prior_predictive(pymc_model):
    """Sample prior predictive distribution."""
    with pymc_model:
        trace_prior = pm.sample_prior_predictive()

    return trace_prior


def sample_prior_fccs(cobra_model):
    """Extract flux control coefficients from prior predictive distribution."""

    # Get prior traces froim unconditional pymc model
    pymc_model = build_unconditional_pymc_model(cobra_model)
    trace_prior = sample_prior_predictive(pymc_model)

    # Get flux star
    v_star = util.calculate_v_star(cobra_model)

    # Get full elasticity matrix
    Ex = util.create_elasticity_matrix(cobra_model)

    # Get external elasticity matrix
    Ey = util.create_Ey_matrix(cobra_model)

    # Get stoichiometric matrix
    N = create_stoichiometric_matrix(cobra_model, dtype='int')

    # Creaet linlog model
    ll = linlog_model.LinLogLeastNorm(N, Ex, Ey, v_star, driver='gelsy')

    fcc_prior = np.array([ll.flux_control_coefficient(Ex=ex) for ex in trace_prior.prior['Ex'][0].to_numpy()])

    return fcc_prior
