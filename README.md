# EBM 

This is the `python` package for implementing [Event Based Models for Disease Progression](https://ebmbook.vercel.app/). 

## Installation

```bash
pip install alabEBM
```

## Generate Random Data

```py
from alabEBM import generate, get_params_path
import os

# Get path to default parameters
params_file = get_params_path()

# Generate data using default parameters
S_ordering = [
    'HIP-FCI', 'PCC-FCI', 'AB', 'P-Tau', 'MMSE', 'ADAS',
    'HIP-GMI', 'AVLT-Sum', 'FUS-GMI', 'FUS-FCI'
]

generate(
    S_ordering=S_ordering,
    real_theta_phi_file=params_file,  # Use default parameters
    js = [50, 100], # Number of participants
    rs = [0.1, 0.5], # Percentage of non-diseased participants
    num_of_datasets_per_combination=2,
    output_dir='my_data'
)
```

## Run MCMC Algorithms 

```py
from alabEBM import run_ebm
from alabEBM.data import get_sample_data_path
import os

print("Current Working Directory:", os.getcwd())

for algorithm in ['soft_kmeans', 'conjugate_priors', 'hard_kmeans']:
    results = run_ebm(
        data_file=get_sample_data_path('25|50_10.csv'),  # Use the path helper
        algorithm=algorithm,
        n_iter=2000,
        n_shuffle=2,
        burn_in=1000,
        thinning=20,
    )
```

## Features

- Multiple MCMC algorithms:
    - Conjugate Priors
    - Hard K-means
    - Soft K-means

- Data generation utilities
- Extensive logging


