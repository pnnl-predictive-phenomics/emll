""" Test file for Antimony to Cobra conversion """

import pytest
import numpy as np
import tellurium as te
import cobra

from emll.util import ant_to_cobra

def compare_stoich_matrices(antimony_path):
    r = te.loada(antimony_path)
    N_te = r.getFullStoichiometryMatrix()
  
    model = cobra.io.read_sbml_model(ant_to_cobra(antimony_path) + '.xml')
    N_cobra = cobra.util.create_stoichiometric_matrix(model)

    assert N_te.shape == N_cobra.shape
    assert (N_te==N_cobra).all()

