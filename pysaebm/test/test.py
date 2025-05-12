from pysaebm import run_ebm
from pysaebm.data import get_sample_data_path, get_params_path
from pysaebm.utils.runners import extract_fname
import os
import json 

cwd = os.getcwd()
print("Current Working Directory:", cwd)
data_dir = f"{cwd}/pysaebm/test/my_data"
data_files = os.listdir(data_dir) 

params_file = get_params_path()

OUTPUT_DIR = 'algo_results'

# Load parameters
with open(params_file) as f:
    biomarker_params = json.load(f)

with open(f"{cwd}/pysaebm/test/true_order_and_stages.json", "r") as f:
    true_order_and_stages = json.load(f)

for algorithm in ['hard_kmeans', 'mle', 'conjugate_priors', 'em', 'kde']:
# for algorithm in ['conjugate_priors']:
    for data_file in data_files:
        fname = data_file.replace('.csv', '')
        true_order_dict = true_order_and_stages[fname]['true_order']
        true_stages = true_order_and_stages[fname]['true_stages']
        results = run_ebm(
            data_file= os.path.join(data_dir, data_file),
            algorithm=algorithm,
            output_dir=OUTPUT_DIR,
            n_iter=200,
            n_shuffle=2,
            burn_in=10,
            thinning=5,
            true_order_dict=true_order_dict,
            true_stages = true_stages,
            skip_heatmap=False,
            skip_traceplot=False
        )