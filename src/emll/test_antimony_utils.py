""" Test file for Antimony to Cobra conversion """

import pytest
import numpy as np
import tellurium as te
import cobra

import logging
logging.getLogger("cobra").setLevel(logging.ERROR)

# from emll.util import ant_to_cobra
from util import ant_to_cobra

@pytest.mark.parametrize("antimony_path", [
    "emll/src/emll/test_models/unlabeled18.ant",
])

def test_antimony_to_cobra_conversion(antimony_path):
    """
    Compares the stoichiometric matrices of original antimony file 
    and cobra-compatible sbml file to see if they are the same.
    
    input: path to original antimony file (non-cobra-compatible)
    """
    r = te.loada(antimony_path)
    N_te = r.getFullStoichiometryMatrix()
    
    file_name = antimony_path.split('.')[0]
    model = cobra.io.read_sbml_model(file_name + '_cobra.xml')
    N_cobra = cobra.util.create_stoichiometric_matrix(model)

    assert N_te.shape == N_cobra.shape
    assert (N_te==N_cobra).all()

