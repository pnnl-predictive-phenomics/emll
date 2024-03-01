"""Code to create preprocessed dataframes for BMCA.

Preprocessing tasks include generating required data from models when experimental data from the user is not provided (i.e. no data cases).
"""

import cobra
import pandas as pd


def get_model(model_path):
    # Load model by type
    if model_path.endswith('.json'):
        model = cobra.io.load_json_model(model_path)
    elif model_path.endswith('.xml') or model_path.endswith('.sbml'):
        model = cobra.io.read_sbml_model(model_path)
    else:
        raise ValueError('Model must be in .json or .xml format')

    return model


# TODO: Create fixtures to run each test for bad and clean versions of input data filenames

def get_metabolomics_data(metab_data_path=None, index_col=0, header=0, model_path=None):
    if metab_data_path is not None:
        if metab_data_path.endswith('.csv'):
            metab_data = pd.read_csv(metab_data_path, index_col=index_col, header=header)
        elif metab_data_path.endswith('.xls') or metab_data_path.endswith('.xlsx'):
            metab_data = pd.read_excel(metab_data_path, index_col=index_col, header=header)
        else:
            raise ValueError('Metabolomics data must be in .csv, .xls, or .xlsx format')
        return metab_data
    else:
        return None


def get_transcriptomics_data(transcript_data_path=None, index_col=0, header=0, model_path=None):
    if transcript_data_path is not None:
        if transcript_data_path.endswith('.csv'):
            transcript_data = pd.read_csv(transcript_data_path, index_col=index_col, header=header)
        elif transcript_data_path.endswith('.xls') or transcript_data_path.endswith('.xlsx'):
            transcript_data = pd.read_excel(transcript_data_path, index_col=index_col, header=header)
        else:
            raise ValueError('Transcriptomics data must be in .csv, .xls, or .xlsx format')
        return transcript_data
    else:
        return None


