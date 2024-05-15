# Code to run the ADVI inference with a near-genome scale model and relative
# omics data.

# So I've found that for certain hardware (the intel chips on the cluster here,
# for instance) the intel python and mkl-numpy are about 2x as fast as the
# openblas versions. You can delete a bunch of this stuff if it doesn't work
# for you. This example is a lot slower than some of the other ones though, but
# I guess that's expected

import os
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as T
import cobra
import emll
from emll.util import initialize_elasticity

os.environ["MKL_THREADING_LAYER"] = "GNU"

# Load model and data
model = cobra.io.load_yaml_model("data/jol2012.yaml")

rxn_compartments = [r.compartments if "e" not in r.compartments else "t" for r in model.reactions]

rxn_compartments[model.reactions.index("SUCCt2r")] = "c"
rxn_compartments[model.reactions.index("ACt2r")] = "c"

for rxn in model.exchanges:
    rxn_compartments[model.reactions.index(rxn)] = "t"

met_compartments = [m.compartment for m in model.metabolites]

flux_ref = pd.read_csv("data/v_star.csv", header=None, index_col=0)[1]

# type_norm|unorm_obs|unobs|ss_df|tensor

metabolites = pd.read_csv("data/metabolite_concentrations.csv", index_col=0)
fluxes = pd.read_csv("data/boundary_fluxes.csv", index_col=0)
enzymes = pd.read_csv("data/enzyme_measurements.csv", index_col=0)

# Reindex arrays to have the same column ordering
to_consider = fluxes.columns
fluxes = fluxes.loc[:, to_consider]
metabolites = metabolites.loc[:, to_consider]
enzymes = enzymes.loc[:, to_consider]

n_exp = len(to_consider) - 1
ref_state = "P0.11"

metabolites_norm = (metabolites.subtract(metabolites["P0.11"], 0) * np.log(2)).T
enzymes_norm = (2 ** enzymes.subtract(enzymes["P0.11"], 0)).T

# To calculate vn, we have to merge in the flux_ref series and do some
# calculations.
flux_ref_df = pd.DataFrame(flux_ref).reset_index().rename(columns={0: "id", 1: "flux"})
fluxes_merge = fluxes.merge(flux_ref_df, left_index=True, right_on="id").set_index("id")
fluxes_norm = fluxes_merge.divide(fluxes_merge.flux, axis=0).drop("flux", axis=1).T

# Drop reference state
fluxes_norm = fluxes_norm.drop(ref_state)
metabolites_norm = metabolites_norm.drop(ref_state)
enzymes_norm = enzymes_norm.drop(ref_state)

# Get indexes for measured values
metabolites_inds = np.array([model.metabolites.index(met) for met in metabolites_norm.columns])
enzymes_inds = np.array([model.reactions.index(rxn) for rxn in enzymes_norm.columns])
fluxes_inds = np.array([model.reactions.index(rxn) for rxn in fluxes_norm.columns])

enzymes_laplace_inds = []
enzymes_zero_inds = []

for i, rxn in enumerate(model.reactions):
    if rxn.id not in enzymes_norm.columns:
        if ("e" not in rxn.compartments) and (len(rxn.compartments) == 1):
            enzymes_laplace_inds += [i]
        else:
            enzymes_zero_inds += [i]

enzymes_laplace_inds = np.array(enzymes_laplace_inds)
enzymes_zero_inds = np.array(enzymes_zero_inds)
enzymes_indexer = np.hstack([enzymes_inds, enzymes_laplace_inds, enzymes_zero_inds]).argsort()

stoichiometric_matrix = cobra.util.create_stoichiometric_matrix(model)
# internal and external elasticity matrix
elasticity_matrix = emll.util.create_elasticity_matrix(model)
external_elasticity_matrix = emll.util.create_Ey_matrix(model) 

elasticity_matrix *= 0.1 + 0.8 * np.random.rand(*elasticity_matrix.shape)

lin_log_model = emll.LinLogLeastNorm(stoichiometric_matrix, elasticity_matrix, external_elasticity_matrix, flux_ref.values, driver="gelsy")

np.random.seed(1)


# Define the probability model

with pm.Model() as pymc_model:
    # Priors on elasticity values (rename) - internal
    elasticity_matrix_tensor = pm.Deterministic(
        "Ex",
        initialize_elasticity(
            lin_log_model.N,
            b=0.01,
            sigma=1,
            alpha=None,
            m_compartments=met_compartments,
            r_compartments=rxn_compartments,
        ),
    )

    # remove 'matrix' from name
    external_elasticity_matrix_tensor = T.as_tensor_variable(external_elasticity_matrix) 

    # use obs vs unobs
    enzymes_measured = pm.Normal("log_e_measured", mu=np.log(enzymes_norm), sigma=0.2, shape=(n_exp, len(enzymes_inds)))
    enzymes_unmeasured = pm.Laplace("log_e_unmeasured", mu=0, b=0.1, shape=(n_exp, len(enzymes_laplace_inds)))
    log_enzymes_norm_tensor = T.concatenate(
        [enzymes_measured, enzymes_unmeasured, T.zeros((n_exp, len(enzymes_zero_inds)))], axis=1
    )[:, enzymes_indexer]

    pm.Deterministic("log_en_t", log_enzymes_norm_tensor)

    # Priors on external concentrations
    external_metabolites_measured = pm.Normal(
        "yn_t", mu=0, sigma=10, shape=(n_exp, lin_log_model.ny), initval=0.1 * np.random.randn(n_exp, lin_log_model.ny)
    )

    # internal and norm
    metabolites_ss, fluxes_norm_ss = lin_log_model.steady_state_pytensor(elasticity_matrix_tensor, external_elasticity_matrix_tensor, T.exp(log_enzymes_norm_tensor), external_metabolites_measured)
    pm.Deterministic("chi_ss", metabolites_ss)
    pm.Deterministic("vn_ss", fluxes_norm_ss)

    log_fluxes_norm_ss = T.log(T.clip(fluxes_norm_ss[:, fluxes_inds], 1e-8, 1e8))
    log_fluxes_norm_ss = T.clip(log_fluxes_norm_ss, -1.5, 1.5)

    transformed_metabolites_ss_clip = T.clip(metabolites_ss[:, metabolites_inds], -1.5, 1.5)

    transformed_metabolites_ss_obs = pm.Normal("chi_obs", mu=transformed_metabolites_ss_clip, sigma=0.2, observed=metabolites_norm.clip(lower=-1.5, upper=1.5))
    log_fluxes_norm_obs = pm.Normal(
        "vn_obs", mu=log_fluxes_norm_ss, sigma=0.1, observed=np.log(fluxes_norm).clip(lower=-1.5, upper=1.5)
    )


if __name__ == "__main__":
    with pymc_model:
        approx = pm.ADVI()
        hist = approx.fit(
            n=40000,
            obj_optimizer=pm.adagrad_window(learning_rate=0.005),
            total_grad_norm_constraint=100,
        )

        # trace = hist.sample(500)
        # ppc = pm.sample_ppc(trace)

    import gzip
    import cloudpickle

    with gzip.open("data/hackett_advi.pgz", "wb") as f:
        cloudpickle.dump(
            {
                "approx": approx,
                "hist": hist,
            },
            f,
        )
