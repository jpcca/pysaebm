import os
import pathlib

def get_data_dir():
    """Get absolute path to the data directory"""
    return os.path.dirname(os.path.abspath(__file__))

def get_sample_data_path(filename):
    """Returns path to a specific sample file"""
    return os.path.join(get_data_dir(), 'samples', filename)

def get_params_path():
    """Returns path to params.json"""
    return os.path.join(get_data_dir(), 'params.json')

# def get_biomarker_order_path():
#     """Returns path to biomarker_order.json"""
#     return os.path.join(get_data_dir(), 'biomarker_order.json')