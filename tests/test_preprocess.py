""" Tests for preprocessing BMCA required data"""

import pytest
import cobra
import pandas as pd
from emll import preprocess


# Define global variables for filenames of testing cases from Hackett example
MODEL = "notebooks/data/jol2012.yaml"
METAB_DATA = "notebooks/data/metabolite_concentrations.csv"
METAB_DATA_CLEAN = "notebooks/data/metabolite_concentrations_clean.csv"
METAB_ROWS = "notebooks/data/metab_rownames.csv"
METAB_COLS = "notebooks/data/metab_colnames.csv"


# Test loading model: using Hackett model as example
def test_get_model():
    model = preprocess.get_model(MODEL)
    assert isinstance(model, cobra.core.model.Model)


# Test loading metabolomics data if no data/pathname is given
def test_get_metabolomics_no_data():
    data = preprocess.get_metabolomics_data()
    assert isinstance(data, None)


# Test loading metabolomics data from file
def test_get_metabolomics_data(fname):
    data = preprocess.get_metabolomics_data(fname)
    assert isinstance(data, pd.DataFrame)


# Test capturing index/rownames from metabolomics data
def test_get_metabolomics_index():
    data = preprocess.get_metabolomics_data(METAB_DATA)
    row_names = pd.read_csv(METAB_ROWS, index_col=0)
    assert isinstance(data.index, pd.MultiIndex)
    assert [c for c in data.index] == row_names


# Test capturing column names from metabolomics data
def test_get_metabolomics_columns():
    data = preprocess.get_metabolomics_data(METAB_DATA)
    col_names = pd.read_csv(METAB_COLS, index_col=0)
    assert isinstance(data.columns, pd.MultiIndex)
    assert [c for c in data.columns] == col_names


# Test if data is in the correct format
def test_get_metabolomics_data_format():
    data = preprocess.get_metabolomics_data(METAB_DATA)
    assert isinstance(data, pd.DataFrame)