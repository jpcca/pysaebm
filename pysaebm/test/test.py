from pysaebm import run_ebm
import os
import json 

cwd = os.getcwd()
print("Current Working Directory:", cwd)
data_dir = f"{cwd}/pysaebm/test/my_data"
data_files = os.listdir(data_dir) 
# print(data_files)

OUTPUT_DIR = 'algo_results'

with open(f"{cwd}/pysaebm/test/true_order_and_stages.json", "r") as f:
    true_order_and_stages = json.load(f)

# for algorithm in ['kde', 'conjugate_priors', "em", 'mle', 'hard_kmeans']:
for algorithm in ['conjugate_priors']:
    for data_file in data_files:
        # print(data_file)
        fname = data_file.replace('.csv', '')
        true_order_dict = true_order_and_stages[fname]['true_order']
        true_stages = true_order_and_stages[fname]['true_stages']
        results = run_ebm(
            algorithm=algorithm,
            data_file= os.path.join(data_dir, data_file),
            output_dir=OUTPUT_DIR,
            n_iter=1000,
            n_shuffle=2,
            burn_in=50,
            thinning=1,
            true_order_dict=true_order_dict,
            true_stages = true_stages,
            skip_heatmap=True,
            skip_traceplot=True,
            seed = 53,
            save_results=True,
            save_details=False,
            save_stage_post=False,
            save_theta_phi=False,
            max_iter_staging=50,
        )