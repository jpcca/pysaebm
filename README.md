# EBM 

This is the `python` package for implementing EBM. 

```py
pip install alabEBM
```

## Usage

To generate random data:

```py
from alabEBM import generate
import numpy as np 
S_ordering = np.array([
        'HIP-FCI', 'PCC-FCI', 'AB', 'P-Tau', 'MMSE', 'ADAS', 
        'HIP-GMI', 'AVLT-Sum', 'FUS-GMI', 'FUS-FCI'
    ])

real_theta_phi_file = '../alabEBM/data/real_theta_phi.json'

js = [50, 100]
rs = [0.1, 0.5]
num_of_datasets_per_combination = 20

generate(
    S_ordering,
    real_theta_phi_file,
    js,
    rs,
    num_of_datasets_per_combination,
    output_dir = 'data'
)
```

To get results:

```py
from alabEBM import run_hard_kmeans
from alabEBM import run_soft_kmeans
from alabEBM import run_conjugate_priors

data_file = '../alabEBM/data/25|50_10.csv'
n_iter = 20
n_shuffle = 2
burn_in = 2
thinning = 2
heatmap_folder = 'heatmap'
filename = '25_50_10_hk'
temp_result_file = f'results/{filename}.json'

run_hard_kmeans(
    data_file,
    n_iter,
    n_shuffle,
    burn_in,
    thinning,
    heatmap_folder,
    filename,
    temp_result_file,
)

filename = '25_50_10_sk'
temp_result_file = f'results/{filename}.json'
run_soft_kmeans(
    data_file,
    n_iter,
    n_shuffle,
    burn_in,
    thinning,
    heatmap_folder,
    filename,
    temp_result_file,
)

filename = '25_50_10_cp'
temp_result_file = f'results/{filename}.json'
run_conjugate_priors(
    data_file,
    n_iter,
    n_shuffle,
    burn_in,
    thinning,
    heatmap_folder,
    filename,
    temp_result_file,
)
```