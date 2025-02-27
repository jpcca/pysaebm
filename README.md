# EBM 

This is the `python` package for implementing [Event Based Models for Disease Progression](https://ebmbook.vercel.app/). 

## Installation

```bash
pip install alabEBM
```

## Change Log

- 2025-02-26. V 0.3.4. Modified the `shuffle_order` function to ensure full derangement, making convergence faster. 

## Generate Random Data

```py
from alabEBM import generate, get_params_path, get_biomarker_order_path
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
    output_dir='my_data',
    seed = None,
    prefix = None,
    suffix = None,
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

## Input data

The input data should have four columns:

- participant: int
- biomarker: str
- measurement: float
- diseased: bool 

An example is https://raw.githubusercontent.com/hongtaoh/alabEBM/refs/heads/main/alabEBM/tests/my_data/10%7C100_0.csv

The data should be in a [tidy format](https://vita.had.co.nz/papers/tidy-data.pdf), i.e.,

- Each variable is a column. 
- Each observation is a row. 
- Each type of observational unit is a table. 

## Features

- Multiple MCMC algorithms:
    - Conjugate Priors
    - Hard K-means
    - Soft K-means

- Data generation utilities
- Extensive logging


