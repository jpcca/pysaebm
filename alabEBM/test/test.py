from alabebm import run_ebm
from alabebm.data import get_sample_data_path, get_params_path
from alabebm.utils.runners import extract_fname
import os
import json 

cwd = os.getcwd()
print("Current Working Directory:", cwd)
data_dir = f"{cwd}/alabebm/test/my_data"
data_files = os.listdir(data_dir) 

params_file = get_params_path()

# Load parameters
with open(params_file) as f:
    biomarker_params = json.load(f)

correct_ordering = {}
for bm, bm_params in biomarker_params.items():
    correct_ordering[bm] = bm_params['order']

for algorithm in ['hard_kmeans', 'mle', 'conjugate_priors', 'em']:
# for algorithm in ['hard_kmeans']:
    for data_file in data_files:
        results = run_ebm(
            data_file= os.path.join(data_dir, data_file),
            algorithm=algorithm,
            n_iter=2000,
            n_shuffle=2,
            burn_in=1000,
            thinning=10,
            correct_ordering=correct_ordering
        )