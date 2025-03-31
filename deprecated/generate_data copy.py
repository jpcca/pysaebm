from typing import List, Optional, Tuple, Dict
import json 
import pandas as pd 
import numpy as np 
import os 
import scipy.stats as stats
from collections import Counter 

###############################################################################
#                           Discrete Stage Model                             #
###############################################################################

def generate_data_discrete(
    n_participants: int,
    biomarker_order: Dict[str, int],
    real_theta_phi_file: str,
    healthy_ratio: float,
    output_dir: str,
    m: int,  # combstr_m
    seed: int,
    stage_distribution: Optional[str] = 'dirichlet',
    dirichlet_alpha: Optional[List[float]] = None,
    prefix: Optional[str] = None,  # Optional prefix
    suffix: Optional[str] = None,   # Optional suffix,
    keep_all_cols: Optional[bool] = False 
) -> pd.DataFrame:
    """
    Simulate an Event-Based Model (EBM) for disease progression.

    Args:
    n_participants (int): Number of participants.
    biomarker_order (Dict[str, int]): Biomarker names and their orders
        in which each of them get affected by the disease.
    real_theta_phi_file (str): Directory of a JSON file which contains 
        theta and phi values for all biomarkers.
        See real_theta_phi.json for example format.
    output_dir (str): Directory where output files will be saved.
    healthy_ratio (float): Proportion of healthy participants out of n_participants.
    seed (int): Seed for the random number generator for reproducibility.
    stage_distribution (Optional[str]), either 'uniform' or 'dirichlet'
    dirichlet_alpha (Optional[List[float]]): the dirichlet distribution alpha vector. Default to be none
    prefix (Optional[str]): Optional prefix of filename
    suffix (Optional[str]): Optional suffix of filename
    keep_all_cols (Optional[bool]): if False, drop ['k_j', 'S_n', 'affected_or_not']

    Returns:
    pd.DataFrame: A DataFrame with columns 'participant', "biomarker", 'measurement', 
        'diseased', with or without ['k_j', 'S_n', 'affected_or_not']
    """
    # Parameter validation
    assert n_participants > 0, "Number of participants must be greater than 0."
    assert 0 <= healthy_ratio <= 1, "Healthy ratio must be between 0 and 1."
    assert stage_distribution in ['dirichlet', 'uniform'], "stage_distribution must be either dirichlet or uniform!"

    # # Change dict to list 
    # biomarker_order = dict(sorted(biomarker_order.items(), key=lambda item:item[1]))
    # ordered_biomarkers = list(biomarker_order.keys())

    # Set the seed for numpy's random number generator
    rng = np.random.default_rng(seed)

    # Load theta and phi values from the JSON file
    try:
        with open(real_theta_phi_file) as f:
            real_theta_phi = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {real_theta_phi_file} not found")
    except json.JSONDecodeError:
        raise ValueError(
            f"File {real_theta_phi_file} is not a valid JSON file.")
            
    n_biomarkers = len(ordered_biomarkers)
    n_stages = n_biomarkers + 1
    max_stage = n_biomarkers
    n_diseased_stages = n_biomarkers

    n_healthy = int(n_participants * healthy_ratio)
    n_diseased = int(n_participants - n_healthy)

    if n_diseased > 0:
        if stage_distribution == "uniform":
            diseased_stages = rng.choice(np.arange(1, max_stage + 1), size=n_diseased)
        elif stage_distribution == "dirichlet":
            dirichlet_alpha = dirichlet_alpha or [1.0] * max_stage
            stage_probs = rng.dirichlet(dirichlet_alpha)
            counts = rng.multinomial(n_diseased, stage_probs)
            diseased_stages = np.repeat(np.arange(1, max_stage + 1), counts)
    else:
        diseased_stages = np.array([], dtype=int)


    all_kjs = np.concatenate([np.zeros(n_healthy), diseased_stages])
    is_diseased = all_kjs > 0 

    # Shuffle participants
    shuffle_idx = rng.permutation(n_participants)
    all_kjs = all_kjs[shuffle_idx]
    is_diseased = is_diseased[shuffle_idx]

    # # Generate disease stages
    # kjs = np.concatenate([np.zeros(n_healthy, dtype=int), diseased_stages])
    # # shuffle so that it's not 0s first and then disease stages but all random
    # rng.shuffle(kjs)

    # ================================================================
    # Biomarker generation (same for all paradigms)
    # ================================================================

    data = []
    for participant_id, (k_j, is_diseased) in enumerate(zip(all_kjs, is_diseased)):
        for biomarker in biomarkers:
            bm_params = params[biomarker]
            xi_i = biomarker_order[biomarker]
            
            # Base noise (β_j,i) from phi distribution
            beta_ji = rng.normal(bm_params["phi_mean"], bm_params["phi_std"])
            
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
                    "S_n": xi_i,
                    "measurement": measurement,
                    "diseased": is_diseased,
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



    # # Initiate biomarker measurement matrix (J participants x N biomarkers) with None
    # # X = np.full((n_participants, n_biomarkers), None, dtype=object)
    # X = np.empty((n_participants, n_biomarkers), dtype=object)

    # # Create distributions for each biomarker
    # theta_dist = {biomarker: stats.norm(
    #     real_theta_phi[biomarker]['theta_mean'],
    #     real_theta_phi[biomarker]['theta_std']
    # ) for biomarker in ordered_biomarkers}

    # phi_dist = {biomarker: stats.norm(
    #     real_theta_phi[biomarker]['phi_mean'],
    #     real_theta_phi[biomarker]['phi_std']
    # ) for biomarker in ordered_biomarkers}

    # # Populate the matrix with biomarker measurements
    # for j in range(n_participants):
    #     for n, biomarker in enumerate(ordered_biomarkers):
    #         # because for each j, we generate X[j, n] in the order of ordered_biomarkers,
    #         # the final dataset will have this ordering as well.
    #         k_j = kjs[j]
    #         S_n = n + 1

    #         # Assign biomarker values based on the participant's disease stage
    #         # affected, or not_affected, is regarding the biomarker, not the participant
    #         if k_j >= 1:
    #             if k_j >= S_n:
    #                 # rvs() is affected by np.random()
    #                 X[j, n] = (
    #                     j, biomarker, theta_dist[biomarker].rvs(random_state=rng), k_j, S_n, 'affected')
    #             else:
    #                 X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
    #                            k_j, S_n, 'not_affected')
    #         # if the participant is healthy
    #         else:
    #             X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
    #                        k_j, S_n, 'not_affected')

    # df = pd.DataFrame(X, columns=ordered_biomarkers)
    # # make this dataframe wide to long
    # df_long = df.melt(var_name="Biomarker", value_name="Value")
    # data = df_long['Value'].apply(pd.Series)
    # data.columns = ['participant', "biomarker",
    #                 'measurement', 'k_j', 'S_n', 'affected_or_not']

    # # biomarker_name_change_dic = dict(
    # #     zip(ordered_biomarkers, range(1, n_biomarkers + 1)))
    # data['diseased'] = data.apply(lambda row: row.k_j > 0, axis=1)
    # if not keep_all_cols:
    #     data.drop(['k_j', 'S_n', 'affected_or_not'], axis=1, inplace=True)
    # # data['biomarker'] = data.apply(
    # #     lambda row: f"{row.biomarker} ({biomarker_name_change_dic[row.biomarker]})", axis=1)

    filename = f"{int(healthy_ratio*n_participants)}_{n_participants}_{m}"
    if prefix:
        filename = f"{prefix}_{filename}"
    if suffix:
        filename = f"{filename}_{suffix}"
    
    output_path = os.path.join(output_dir, f"{filename}.csv")
    data.to_csv(output_path, index=False)
    print("Data generation done! Output saved to:", filename)
    return data

def generate_discrete(
    biomarker_order: Dict[str, int],
    real_theta_phi_file: str,
    js: List[int],
    rs: List[float],
    num_of_datasets_per_combination: int,
    output_dir: str,
    m: int,
    seed: Optional[int] = None,
    stage_distribution: Optional[str] = 'dirichlet',
    dirichlet_alpha: Optional[List[float]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    keep_all_cols: Optional[bool] = False 
):
    """
    Generates datasets for multiple combinations of participants, healthy ratios, and datasets.

    Args:
    biomarker_order (Dict[str, int]): Biomarker names and their orders
    real_theta_phi_file (str): Path to the JSON file containing theta and phi values.
    js (List[int]): List of numbers of participants.
    rs (List[float]): List of healthy ratios.
    num_of_datasets_per_combination (int): Number of datasets to generate per combination.
    output_dir (str): Directory to save the generated datasets.
    seed (Optional[int]): Global seed for reproducibility. If None, a random seed is used.
    stage_distribution (Optional[str]), either 'uniform' or 'dirichlet'
    dirichlet_alpha (Optional[List[float]]): the dirichlet distribution alpha vector. Default to be none
    prefix (Optional[str]): Optional prefix of filename
    suffix (Optional[str]): Optional suffix of filename
    keep_all_cols (Optional[bool]): if False, drop ['k_j', 'S_n', 'affected_or_not']
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if seed is None:
        seed = np.random.SeedSequence().entropy 
    rng = np.random.default_rng(seed)

    for j in js:
        for r in rs:
            for m in range(num_of_datasets_per_combination):
                sub_seed = rng.integers(0, 1_000_000)
                generate_data_discrete(
                    n_participants=j,
                    biomarker_order=biomarker_order,
                    real_theta_phi_file=real_theta_phi_file,
                    healthy_ratio=r,
                    output_dir=output_dir,
                    m=m,
                    seed=sub_seed,
                    stage_distribution=stage_distribution,
                    dirichlet_alpha = dirichlet_alpha,
                    prefix=prefix,
                    suffix=suffix,
                    keep_all_cols = keep_all_cols
                )
    print(f"Data generation complete. Files saved in {output_dir}/")

###############################################################################
#                                  Sigmoid Model                              #
###############################################################################

def generate_data_sigmoid(
    n_participants: int,
    biomarker_order: Dict[str, int],
    sigmoid_params_file: str,
    healthy_ratio: float,
    output_dir: str,
    m: int,
    seed: int,
    stage_distribution: str = "discrete_uniform",
    dirichlet_alpha: Optional[List[float]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    keep_all_cols: Optional[bool] = False 
) -> pd.DataFrame:
    """
    Unified generator for continuous/discrete stage models.
    
    Args:
        stage_distribution: One of [
            "continuous_uniform", "continuous_beta",
            "discrete_uniform", "discrete_dirichlet"
        ]
    """
    # Validate inputs
    assert stage_distribution in [
        "continuous_uniform", "continuous_beta",
        "discrete_uniform", "discrete_dirichlet"
    ], f"Invalid stage distribution: {stage_distribution}"

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Load parameters
    with open(sigmoid_params_file) as f:
        params = json.load(f)

    biomarkers = sorted(biomarker_order.keys(), key=lambda x: biomarker_order[x])
    n_biomarkers = len(biomarkers)
    max_stage = n_biomarkers
    disease_stages = np.arange(1, max_stage + 1)

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
            beta_params = params.get("beta_params", {"alpha": 2, "beta": 5})
            raw = rng.beta(beta_params["alpha"], beta_params["beta"], size=n_diseased)
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
    is_diseased = all_kjs > 0 

    # Shuffle participants
    shuffle_idx = rng.permutation(n_participants)
    all_kjs = all_kjs[shuffle_idx]
    is_diseased = is_diseased[shuffle_idx]

    # ================================================================
    # Biomarker generation (same for all paradigms)
    # ================================================================
    data = []
    for participant_id, (k_j, is_diseased) in enumerate(zip(all_kjs, is_diseased)):
        for biomarker in biomarkers:
            bm_params = params[biomarker]
            xi_i = biomarker_order[biomarker]
            
            # Base noise (β_j,i) from phi distribution
            beta_ji = rng.normal(bm_params["phi_mean"], bm_params["phi_std"])
            
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
                    "S_n": xi_i,
                    "measurement": measurement,
                    "diseased": is_diseased,
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
    filename = f"{int(healthy_ratio*n_participants)}_{n_participants}_{m}"
    # filename = f"{stage_distribution}_n{n_participants}_h{healthy_ratio}_s{seed}"
    if prefix: filename = f"{prefix}_{filename}"
    if suffix: filename = f"{filename}_{suffix}"
    df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)
    return df

def generate_sigmoid(
    biomarker_order: Dict[str, int],
    sigmoid_params_file: str,
    js: List[int],
    rs: List[float],
    num_of_datasets_per_combination: int,
    output_dir: str,
    m: int,
    seed: Optional[int] = None,
    stage_distribution: Optional[str] = 'discrete_uniform',
    dirichlet_alpha: Optional[List[float]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    keep_all_cols: Optional[bool] = False 
):
    """
    Generates datasets for multiple combinations of participants, healthy ratios, and datasets.

    Args:
    biomarker_order (Dict[str, int]): Biomarker names and their orders
    sigmoid_params_file (str): Path to the JSON file containing parameters.
    js (List[int]): List of numbers of participants.
    rs (List[float]): List of healthy ratios.
    num_of_datasets_per_combination (int): Number of datasets to generate per combination.
    output_dir (str): Directory to save the generated datasets.
    seed (Optional[int]): Global seed for reproducibility. If None, a random seed is used.
    stage_distribution (Optional[str]), chooose from "continuous_uniform", "continuous_beta",
        "discrete_uniform", and "discrete_dirichlet"
    dirichlet_alpha (Optional[List[float]]): the dirichlet distribution alpha vector. Default to be None
    prefix (Optional[str]): Optional prefix of filename
    suffix (Optional[str]): Optional suffix of filename
    keep_all_cols (Optional[bool]): if False, drop ['k_j', 'S_n', 'affected']
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if seed is None:
        seed = np.random.SeedSequence().entropy 
    rng = np.random.default_rng(seed)

    for j in js:
        for r in rs:
            for m in range(num_of_datasets_per_combination):
                sub_seed = rng.integers(0, 1_000_000)
                generate_data_sigmoid(
                    n_participants=j,
                    biomarker_order=biomarker_order,
                    sigmoid_params_file = sigmoid_params_file,
                    healthy_ratio=r,
                    output_dir=output_dir,
                    m=m,
                    seed=sub_seed,
                    stage_distribution=stage_distribution,
                    dirichlet_alpha = dirichlet_alpha,
                    prefix=prefix,
                    suffix=suffix,
                    keep_all_cols = keep_all_cols
                )
    print(f"Data generation complete. Files saved in {output_dir}/")