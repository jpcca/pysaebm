from alabebm import run_ebm
from alabebm.data import get_sample_data_path, get_biomarker_order_path
import os
import json 

cwd = os.getcwd()
print("Current Working Directory:", cwd)
data_dir = f"{cwd}/alabebm/test/my_data"
data_files = os.listdir(data_dir) 

# Get path to biomarker_order
biomarker_order_json = get_biomarker_order_path()

with open(biomarker_order_json, 'r') as file:
    biomarker_order = json.load(file)

# for algorithm in ['hard_kmeans', 'soft_kmeans', 'conjugate_priors']:
for algorithm in ['conjugate_priors']:
    for data_file in data_files:
        results = run_ebm(
            data_file= f"{data_dir}/{data_file}",
            # data_file=get_sample_data_path('10|100_0.csv'),  # Use the path helper
            algorithm=algorithm,
            n_iter=2000,
            n_shuffle=2,
            burn_in=1000,
            thinning=20,
            correct_ordering=biomarker_order
        )