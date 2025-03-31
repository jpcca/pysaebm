from typing import List, Optional, Tuple, Dict
import json 
import pandas as pd 
import numpy as np 
import os 
import scipy.stats as stats
from collections import Counter 

def generate_data(
    framework: str,
    params_file: str,
    n_participants: int,
    healthy_ratio: float,
    output_dir: str,
    m: int,  # combstr_m
    seed: int,
    stage_distribution: Optional[str],
    dirichlet_alpha: Optional[List[float]],
    beta_dist_alpha: Optional[float],
    beta_dist_beta: Optional[float],
    prefix: Optional[str],
    suffix: Optional[str],
    keep_all_cols: Optional[bool]
) -> pd.DataFrame:
    """
    Simulate an Event-Based Model (EBM) for disease progression.

    Args:

    framework (str): either 'discrete' or 'sigmoid'
    params_file (str): Directory of the params.json 
    n_participants (int): Number of participants.
    healthy_ratio (float): Proportion of healthy participants out of n_participants.
    output_dir (str): Directory to save the generated datasets.
    m (int): variant of this combination
    seed (Optional[int]): Global seed for reproducibility. If None, a random seed is used.
    stage_distribution (Optional[str]), chooose from "continuous_uniform", "continuous_beta",
        "discrete_uniform", and "discrete_dirichlet"
    dirichlet_alpha (Optional[List[float]]): the dirichlet distribution alpha vector. Default to be None
    beta_dist_alpha (Optional[float] = 2.0): beta distribution for continuous stages generation
    beta_dist_beta (Optional[float] = 5.0): beta distribution for continuous stages generation
    prefix (Optional[str]): Optional prefix of filename
    suffix (Optional[str]): Optional suffix of filename
    keep_all_cols (Optional[bool]): if False, drop ['k_j', 'order', 'affected']

    Returns:
    pd.DataFrame: A DataFrame with columns 'participant', "biomarker", 'measurement', 
        'diseased', with or without ['k_j', 'order', 'affected']
    """
    # Parameter validation
    assert n_participants > 0, "Number of participants must be greater than 0."
    assert 0 <= healthy_ratio <= 1, "Healthy ratio must be between 0 and 1."
    assert framework in ['sigmoid', 'discrete'], "framework must either sigmoid or discrete"
    if framework == 'sigmoid':
        assert stage_distribution in [
            "continuous_uniform", "continuous_beta",
            "discrete_uniform", "discrete_dirichlet"
        ], f"Invalid stage distribution: {stage_distribution}"
    else:
        assert stage_distribution in [
            "discrete_uniform", "discrete_dirichlet"
        ], f"Invalid stage distribution: {stage_distribution}"

    rng = np.random.default_rng(seed)

    # Load parameters
    with open(params_file) as f:
        params = json.load(f)

    # Calculate max_stage from actual biomarker orders
    orders = [v["order"] for v in params.values()]
    max_stage = max(orders)
    biomarkers = list(params.keys())
    n_biomarkers = len(params)

    # Generate disease status
    n_healthy = int(n_participants * healthy_ratio)
    n_diseased = n_participants - n_healthy

    # ================================================================
    # Core generation logic 
    # ================================================================

    if stage_distribution.startswith('continuous'):
        if stage_distribution == 'continuous_uniform':
            kjs = rng.uniform(0, max_stage + 1, size = n_diseased)
        elif stage_distribution == 'continuous_beta':
            # Generate Beta-distributed values on [0, 1]
            raw = rng.beta(beta_dist_alpha, beta_dist_beta, size=n_diseased)
            # Scale to disease timeline [0, N+1)
            kjs = raw * (max_stage + 1)
    else:
        if stage_distribution == "discrete_uniform":
            kjs = rng.choice(np.arange(1, max_stage + 1), size=n_diseased)
        elif stage_distribution == "discrete_dirichlet":
            dirichlet_alpha = dirichlet_alpha or [1.0] * max_stage
            stage_probs = rng.dirichlet(dirichlet_alpha)
            counts = rng.multinomial(n_diseased, stage_probs)
            kjs = np.repeat(np.arange(1, max_stage + 1), counts)
    
    all_kjs = np.concatenate([np.zeros(n_healthy), kjs])
    all_diseased = all_kjs > 0 

    # Shuffle participants
    shuffle_idx = rng.permutation(n_participants)
    all_kjs = all_kjs[shuffle_idx]
    all_diseased = all_diseased[shuffle_idx]

    # print(int(healthy_ratio*n_participants) == n_participants - sum(all_diseased))

    # ================================================================
    # Biomarker generation (same for all paradigms)
    # ================================================================
    data = []
    for participant_id, (k_j, is_diseased) in enumerate(zip(all_kjs, all_diseased)):
        for biomarker in biomarkers:
            bm_params = params[biomarker]
            xi_i = bm_params['order']
            
            # Base noise (β_j,i) from phi distribution
            # Nonaffected biomarker
            beta_ji = rng.normal(bm_params["phi_mean"], bm_params["phi_std"])

            if framework == 'discrete':
                if not is_diseased or k_j < xi_i:
                    measurement = beta_ji
                else:
                    measurement = rng.normal(bm_params['theta_mean'], bm_params['theta_std'])
            else:
                if is_diseased:
                    # Diseased: sigmoid progression + noise
                    R_i = bm_params["R_i"]
                    rho_i = bm_params["rho_i"]
                    sigmoid_term = R_i / (1 + np.exp(-rho_i * (k_j - xi_i)))
                    measurement = sigmoid_term + beta_ji
                else:
                    # Healthy: pure noise (no disease effect)
                    measurement = beta_ji
            
            if keep_all_cols:
                data.append({
                    "participant": participant_id,
                    "biomarker": biomarker,
                    "measurement": measurement,
                    "diseased": is_diseased,
                    "order": xi_i,
                    "k_j": k_j,
                    "affected": k_j >= xi_i
                })
            else:
                data.append({
                    "participant": participant_id,
                    "biomarker": biomarker,
                    "measurement": measurement,
                    "diseased": is_diseased,
                })
    
    # Save to CSV
    df = pd.DataFrame(data)
    filename = f"{int(healthy_ratio*n_participants)}_{n_participants}_{framework}_{stage_distribution}_{m}"
    # filename = f"{stage_distribution}_n{n_participants}_h{healthy_ratio}_s{seed}"
    if prefix: filename = f"{prefix}_{filename}"
    if suffix: filename = f"{filename}_{suffix}"
    df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)
    return df

def generate(
    framework: str = 'discrete', # either 'discrete' or 'sigmoid'
    params_file: str = 'params.json',
    js: List[int] = [50, 200, 500],
    rs: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    num_of_datasets_per_combination: int = 50,
    output_dir: str = 'data',
    seed: Optional[int] = None,
    stage_distribution: Optional[str] = 'discrete_dirichlet',
    dirichlet_alpha: Optional[List[float]] = None,
    beta_dist_alpha: Optional[float] = 2.0, # beta distribution for continuous stages generation
    beta_dist_beta: Optional[float] = 5.0, # beta distribution for continuous stages generation
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    keep_all_cols: Optional[bool] = False 
):
    """
    Generates datasets for multiple combinations of participants, healthy ratios, and datasets.

    Args:
    framework (str): either 'discrete' or 'sigmoid'
    params_file (str): Directory of the params.json 
    js (List[int]): List of numbers of participants.
    rs (List[float]): List of healthy ratios.
    num_of_datasets_per_combination (int): Number of datasets to generate per combination.
    output_dir (str): Directory to save the generated datasets.
    seed (Optional[int]): Global seed for reproducibility. If None, a random seed is used.
    stage_distribution (Optional[str]), chooose from "continuous_uniform", "continuous_beta",
        "discrete_uniform", and "discrete_dirichlet"
    dirichlet_alpha (Optional[List[float]]): the dirichlet distribution alpha vector. Default to be None
    beta_dist_alpha (Optional[float] = 2.0): beta distribution for continuous stages generation
    beta_dist_beta (Optional[float] = 5.0): beta distribution for continuous stages generation
    prefix (Optional[str]): Optional prefix of filename
    suffix (Optional[str]): Optional suffix of filename
    keep_all_cols (Optional[bool]): if False, drop ['k_j', 'order', 'affected']
    """
    # Ensure output directory exists
    # Won't clear the folder if it already exists
    os.makedirs(output_dir, exist_ok=True)

    if seed is None:
        seed = np.random.SeedSequence().entropy 
    rng = np.random.default_rng(seed)

    for j in js:
        for r in rs:
            for variant in range(num_of_datasets_per_combination):
                sub_seed = rng.integers(0, 1_000_000)
                generate_data(
                    framework=framework,
                    params_file=params_file,
                    n_participants=j,
                    healthy_ratio=r,
                    output_dir=output_dir,
                    m=variant,
                    seed=sub_seed,
                    stage_distribution=stage_distribution,
                    dirichlet_alpha = dirichlet_alpha,
                    beta_dist_alpha=beta_dist_alpha,
                    beta_dist_beta=beta_dist_beta,
                    prefix=prefix,
                    suffix=suffix,
                    keep_all_cols = keep_all_cols
                )
    print(f"Data generation complete. Files saved in {output_dir}/")