from alabebm import run_ebm
from alabebm.data import get_sample_data_path, get_biomarker_order_path
from alabebm.utils.runners import extract_fname
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

for algorithm in ['hard_kmeans', 'mle', 'conjugate_priors', 'em']:
    for data_file in data_files:
        results = run_ebm(
            data_file= os.path.join(data_dir, data_file),
            algorithm=algorithm,
            n_iter=200,
            n_shuffle=2,
            burn_in=100,
            thinning=2,
            correct_ordering=biomarker_order
        )