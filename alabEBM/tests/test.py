from alabEBM import run_ebm
from alabEBM.data import get_sample_data_path
import os
import alabEBM

cwd = os.getcwd()
print("Current Working Directory:", cwd)
data_dir = f"{cwd}/alabEBM/tests/my_data"
data_files = os.listdir(data_dir) 

for algorithm in ['hard_kmeans']:
    for data_file in data_files:
        results = run_ebm(
            data_file= f"{data_dir}/{data_file}",
            # data_file=get_sample_data_path('25|50_10.csv'),  # Use the path helper
            algorithm=algorithm,
            n_iter=500,
            n_shuffle=2,
            burn_in=100,
            thinning=10,
        )