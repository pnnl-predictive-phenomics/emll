import pytest
from src.emll import bmca
import cobra
import pandas as pd


# Get hackett data and initialize BMCA object
@pytest.fixture(scope="module")
def hackett_bmca_obj():
    hackett_paths = {
        "model_path": "notebooks/data/data/jol2012.yaml",
        "v_star_path": "notebooks/data/v_star.csv",
        "metabolite_concentrations_path": "notebooks/data/metabolite_concentrations.csv",
        "boundary_fluxes_path": "notebooks/data/boundary_fluxes.csv",
        "enzyme_measurements_path": "notebooks/data/enzyme_measurements.csv",
        "reference_state":  "P0.11",
    }
    bmca_obj = bmca.BMCA(
        model_path = hackett_paths["model_path"],
        metabolite_concentrations_path = hackett_paths["metabolite_concentrations_path"],
        enzyme_measurements_path = hackett_paths["enzyme_measurements_path"],
        reference_state = hackett_paths["reference_state"],
        boundary_fluxes_path = hackett_paths["boundary_fluxes_path"],
        v_star_path = hackett_paths["v_star_path"],
    )
    return bmca_obj


# Test types of initialized BMCA object
def test_init_type(hackett_bmca_obj):
    bmca_obj = hackett_bmca_obj()
    
    assert isinstance(bmca_obj.model, cobra.core.model.Model)
    assert isinstance(bmca_obj.x, pd.DataFrame)
    assert isinstance(bmca_obj.e, pd.DataFrame)
    assert (bmca_obj.reference_state in bmca_obj.x.columns) and (bmca_obj.reference_state in bmca_obj.e.columns) and (bmca_obj.reference_state in bmca_obj.v.columns)
    assert isinstance(bmca_obj.v, pd.DataFrame)

    # Future will remove v_star from initialized object
    assert isinstance(bmca_obj.v_star, pd.DataFrame)


# Test BMCA object initial structure
def test_initial_structure(hackett_bmca_obj):
    bmca_obj = hackett_bmca_obj()
    
    # Check if metabolomics & enzyme measurement AND flux data rows are in the model
    with bmca_obj.model as model:
        metabs = [m.id[:-2] for m in model.metabolites]
        rxns = [r.id for r in model.reactions]
    assert all([m in metabs for m in bmca_obj.x.index.values])
    assert all([r in rxns for r in bmca_obj.e.index.values])
    assert all([r in rxns for r in bmca_obj.v.index.values])

    # Check if metabolomics & enzyme measurements have same # of columns & same column names
    assert bmca_obj.x.columns.is_unique
    assert bmca_obj.e.columns.is_unique
    assert bmca_obj.x.shape[1] == bmca_obj.e.shape[1]
    assert all(bmca_obj.x.columns.values == bmca_obj.e.columns.values)

    # Check if flux dataframes have same # of columns & same column names as metabolomics measurements
    assert bmca_obj.v.columns.is_unique
    assert bmca_obj.v.shape[1] == bmca_obj.x.shape[1]
    assert all(bmca_obj.v.columns.values == bmca_obj.x.columns.values)
   
    # Check that v_star is the referenced column of the flux data
    # assert bmca_obj.v_star.equals(bmca_obj.v[bmca_obj.v[reference_state])



# Run preprocess of BMCA data to test each sub-routine
@pytest.fixture(scope="module")
def preprocessed_bmca_obj(hackett_bmca_obj):
    bmca_obj = hackett_bmca_obj()
    return bmca_obj.preprocess_data()


# Test establishing compartments
def test_establish_compartments(preprocessed_bmca_obj):
    bmca_obj = preprocessed_bmca_obj(hackett_bmca_obj)
    
    # TODO: check r_compartments and m_compartments

    pass

def test_reindexing_bmca_slots(preprocessed_bmca_obj):
    bmca_obj = preprocessed_bmca_obj(hackett_bmca_obj)
    
    # TODO: check x, v, and e slots in bmca_obj for changes
    pass

def test_normalizing_bmca_slots(preprocessed_bmca_obj):
    bmca_obj = preprocessed_bmca_obj(hackett_bmca_obj)
    
    # TODO: check normalization of x, v, and e as xn, vn, and en slots in bmca_obj
    pass

def test_index_measured_bmca_slots(preprocessed_bmca_obj):
    bmca_obj = preprocessed_bmca_obj(hackett_bmca_obj)
    
    # TODO: check x_inds, e_inds, v_inds, e_laplace_inds, e_zero_inds,  slots in bmca_obj
    pass

def test_new_matrices(preprocessed_bmca_obj):
    bmca_obj = preprocessed_bmca_obj(hackett_bmca_obj)
    
    # TODO: check N, Ex, Ey, ll slots in bmca_obj
    pass


# Build pymc model
def pymc_model_bmca_obj(preprocessed_bmca_obj):
    bmca_obj = preprocessed_bmca_obj(hackett_bmca_obj)
    bmca_obj.build_pymc_model()
    
    return bmca_obj


# Test building pymc model
def test_build_pymc_model(pymc_model_bmca_obj, preprocessed_bmca_obj, hackett_bmca_obj):  # ** Do I need to pass hackett_bmca_obj here?
    bmca_obj = pymc_model_bmca_obj(preprocessed_bmca_obj(hackett_bmca_obj))
    
    # TODO: check Ex_t, Ey_t, pymc_model slots in bmca_obj
    pass

# Test running ADVI
def test_run_advi(pymc_model_bmca_obj, preprocessed_bmca_obj, hackett_bmca_obj):
    bmca_obj = pymc_model_bmca_obj(preprocessed_bmca_obj(hackett_bmca_obj))
    approx, hist = emll_model.run_emll()

    # TODO: check approx, hist
    pass
