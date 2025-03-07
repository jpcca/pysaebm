from alabebm import generate, get_params_path, get_biomarker_order_path
import os
import json 

# Get path to default parameters
params_file = get_params_path()

# Get path to biomarker_order
biomarker_order_json = get_biomarker_order_path()

with open(biomarker_order_json, 'r') as file:
    biomarker_order = json.load(file)

generate(
    biomarker_order = biomarker_order,
    real_theta_phi_file=params_file,  # Use default parameters
    js = [50, 100],
    rs = [0.1, 0.5],
    num_of_datasets_per_combination=2,
    output_dir='my_data'
)