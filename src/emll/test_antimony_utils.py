"""Test file for Antimony to Cobra conversion"""

import logging
from pathlib import Path

import cobra
import pytest
import tellurium as te

from emll.util import ant_to_cobra

logging.getLogger("cobra").setLevel(logging.ERROR)

HERE = Path(__file__).parent.resolve()
MODEL = HERE.joinpath("test_models/unlabeled18.ant")


@pytest.mark.parametrize(
    "antimony_path",
    [
        str(MODEL),
    ],
)
def test_antimony_to_cobra_conversion(antimony_path):
    """
    Compares the stoichiometric matrices of original antimony file
    and cobra-compatible sbml file to see if they are the same.

    input: path to original antimony file (non-cobra-compatible)
    """
    ant_to_cobra(antimony_path)

    r = te.loada(antimony_path)
    N_te = r.getFullStoichiometryMatrix()

    file_name = antimony_path.split(".")[0]
    model = cobra.io.read_sbml_model(file_name + "_cobra.xml")
    N_cobra = cobra.util.create_stoichiometric_matrix(model)

    assert N_te.shape == N_cobra.shape
    assert (N_te == N_cobra).all()
