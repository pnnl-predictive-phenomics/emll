import pytest
import emll
import pandas as pd
import scipy as sp
from emll import bmca


# Get wu2004 data and initialize BMCA object
@pytest.fixture(scope="module")
def wu2004_bmca_obj():
    wu2004_paths = {
        "model_path": "tests/test_models/wu2004_model.sbml",
        "reference_state": None,
    }
    bmca_obj = emll.bmca.BMCA(
        model_path = wu2004_paths["model_path"],
        reference_state = wu2004_paths["reference_state"],
    )
    return bmca_obj


# def test_elasticity_matrix(wu2004_bmca_obj):


def test_priorpredictive(wu2004_bmca_obj):
    bmca_obj = wu2004_bmca_obj

    # # Build unconditional probabilistic model
    # Sample flux control coefficients from prior predictive distribution of an unconditional prob model
    fcc_prior = emll.pymc_utils.sample_prior_fccs(bmca_obj.model)

    # Compare prior predictive distribution to wu2004 prior predictive results
    wu_prior_df = pd.read_csv("tests/test_data/expected_wu2004_FCCpriors.csv", index_col=0)
    met_names = wu_prior_df.index
    for m in met_names:
        this_wu_prior = wu_prior_df.loc[m]
        test_result = sp.stats.wilcoxon(x=fcc_prior, y=this_wu_prior)
    



    # assert bmca_obj.prior_predictive.equals(wu_prior_pred)
