# EBM 

This is the `python` package for implementing [Event Based Models for Disease Progression](https://ebmbook.vercel.app/). 

## Installation

```bash
pip install alabebm
```

## Change Log

- 2025-02-26 (V 0.3.4). 
    - Modified the `shuffle_order` function to ensure full derangement, making convergence faster. 
- 2025-03-06 (V 0.4.0)
    - use `pyproject.toml` instead
    - update `conjuage_priors_algo.py`, now without using the auxiliary variable of `participant_stages`. Kept the uncertainties just like in `soft_kmeans_algo.py`. 
- 2025-03-07 (V 0.4.2)
    - Compute `new_ln_likelihood_new_theta_phi` based on `new_theta_phi_estimates`, which is based on `stage_likelihoods_posteriors` that is based on the newly proposed order and previous `theta_phi_estimates`.
    - Update `theta_phi_estimates` with `new_theta_phi_estimates` only if new order is accepted.
    - The fallback theta_phi_estimates is the previous parameters rather than theta_phi_default 
    - `all_accepted_orders.append(current_order_dict.copy())` to make sure the results are not mutated. 
    - Previously I calculated the `new_ln_likelihood` and `stage_likelihoods_posteriors` based on the newly proposed order and previous `theta_phi_estimates`, and directly update theta_phi_estimates whether we accept the new order or not. 
    - Previously, I excluded `copy()` in `all_accepted_orders.append(current_order_dict.copy())`, which is inaccurate. 
- 2025-03-17 (V 0.4.3)
    - Added `skip` and `title_detail` parameter in `save_traceplot` function. 
- 2025-03-18 (V 0.4.4)
    - Add optional horizontal bar indicating upper limit in trace plot. 
- 2025-03-18 (V 0.4.7)
    - Allowed keeping all cols (`keep_all_cols`) in data generation. 
- 2025-03-18 (V 0.4.9)
    - copy `data_we_have` and use `data_we_have.loc[:, 'S_n']` in soft kmeans algo when preprocessing participant and biomarker data.
- 2025-03-20 (V 0.5.1)
    - In hard kmeans, updated `delta = ln_likelihood - current_ln_likelihood`, and in soft kmeans and conjugate priors, made sure I am using `delta = new_ln_likelihood_new_theta_phi - current_ln_likelihood`.
    - In each iteration, use `theta_phi_estimates = theta_phi_default.copy()` first. This means, `stage_likelihoods_posteriors` is based on the default theta_phi, not the previous iteration. 
- 2025-03-21 (V 0.6.0)
    - Integrated all three algorithms to just one file `algorithms/algorithm.py`. 
    - Changed the algorithm name of `soft_kmeans` to `mle` (maximum likelihood estimation)
    - Moved all helper functions from the algorithm script to `utils/data_processing.py`. 
- 2025-03-22 (V 0.7.6)
    - Current state should include both the current accepted order and its associated theta/phi. When updating theta/phi at the start of each iteration, use the current state's theta/phi (1) in the calculation of stage likelihoods and (2) as the fallback if either of the biomarker's clusters is empty or has only one measurement; (3) as the prior mean and variance. 
    - Set `conjugate_priors` as the default algorithm. 
    - (Tried using cluster's mean and var as the prior but the results are not as good as using current state's theta/phi as the prior). 
- 2025-03-24 (V 0.7.8)
    - In heatmap, reorder the biomarkers according to the most likely order. 
    - In `results.json` reorder the biomarker according to their order rather than alphabetically ranked. 
    - Modified `obtain_most_likely_order_dic` so that we assign stages for biomarkers that have the highest probabilities first. 
    - In `results.json`, output the order associated with the highest total log likelihood. Also, calculate the kendall's tau and p values of it and the original order (if provided).

## Generate Random Data

```py
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
    output_dir='my_data',
    seed = None,
    prefix = None,
    suffix = None,
    keep_all_cols = False
)
```

## Run MCMC Algorithms 

```py
from alabebm import run_ebm
from alabebm.data import get_sample_data_path
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

The input data should have at least four columns:

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
    - MLE

- Data generation utilities
- Extensive logging


