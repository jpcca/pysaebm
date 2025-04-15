from typing import List, Optional, Tuple, Dict
import json 
import pandas as pd 
import numpy as np 
import os 
from collections import Counter, defaultdict

experiment_names = [
    "sn_kjOrdinalDM_xnjNormal", # Experiment 1
	"sn_kjOrdinalDM_xnjNonNormal", # Experiment 2
    "sn_kjOrdinalUniform_xnjNormal", # Experiment 3
	"sn_kjOrdinalUniform_xnjNonNormal", # Experiment 4
    "sn_kjContinuousUniform", # Experiment 5
	"sn_kjContinuousBeta", # Experiment 6
    "xiNearNormal_kjContinuousUniform", # Experiment 7
	"xiNearNormal_kjContinuousBeta", # Experiment 8
]

def very_irregular_distribution(biomarker: str, bm_params: dict, state="affected", size=10_0000, rng=None):
    """
    Generate highly irregular, non-normal samples for a given biomarker and state.
    Combines different distributions and transformations to create irregular shapes.
    """
    if rng is None:
        rng = np.random.default_rng()

    mean = bm_params["theta_mean"] if state == "affected" else bm_params["phi_mean"]
    std = bm_params["theta_std"] if state == "affected" else bm_params["phi_std"]

    base = np.zeros(size)
    s1, s2, s3 = np.array_split(np.arange(size), 3)

    # --- Highly non-normal design per biomarker ---
    if biomarker in ["MMSE", "ADAS"]:
        base[s1] = rng.triangular(mean - 2*std, mean - 1.5*std, mean, size=len(s1))
        base[s2] = rng.normal(mean + std, 0.3 * std, size=len(s2))
        base[s3] = rng.exponential(scale=0.7 * std, size=len(s3)) + mean - 0.5 * std

    elif biomarker in ["AB", "P-Tau"]:
        base[s1] = rng.pareto(1.5, size=len(s1)) * std + mean - 2 * std
        base[s2] = rng.uniform(mean - 1.5 * std, mean + 1.5 * std, size=len(s2))
        base[s3] = rng.logistic(loc=mean, scale=std, size=len(s3))

    elif biomarker in ["HIP-FCI", "HIP-GMI"]:
        base[s1] = rng.beta(0.5, 0.5, size=len(s1)) * 4 * std + mean - 2 * std
        base[s2] = rng.exponential(scale=std * 0.4, size=len(s2)) * rng.choice([-1, 1], size=len(s2)) + mean
        base[s3] = rng.normal(mean, std * 0.5, size=len(s3)) + rng.choice([0, std * 2], size=len(s3))

    elif biomarker in ["AVLT-Sum", "PCC-FCI"]:
        base[s1] = rng.gamma(shape=2, scale=0.5 * std, size=len(s1)) + mean - std
        base[s2] = rng.weibull(1.0, size=len(s2)) * std + mean - std
        base[s3] = rng.normal(mean, std * 0.5, size=len(s3)) + rng.choice([-1, 1], size=len(s3)) * std

    elif biomarker == "FUS-GMI":
        raw = rng.standard_cauchy(size=size) * std + mean
        raw += rng.normal(0, 0.2 * std, size=size)
        base = np.clip(raw, mean - 4 * std, mean + 4 * std)

    elif biomarker == "FUS-FCI":
        spike_size = size // 10
        base[:spike_size] = rng.normal(mean, 0.2 * std, size=spike_size)
        base[spike_size:] = rng.logistic(loc=mean + std, scale=2 * std, size=size - spike_size)

    else:
        base = rng.uniform(mean - 2 * std, mean + 2 * std, size=size)

    # --- Add irregular noise and clip ---
    base += rng.normal(0, 0.2 * std, size=size)  # extra randomness
    base = np.clip(base, mean - 5 * std, mean + 5 * std)

    return base

def generate_measurements_kjOrdinal(
    params, event_time_dict, shuffled_biomarkers, experiment_name, all_kjs, 
    all_diseased, keep_all_cols, rng = None
    ):
    if rng is None:
        rng = np.random.default_rng()

    data = []
    # Get many data points for random drawing
    irreg_dict = defaultdict(dict)
    if "xnjNonNormal" in experiment_name:
        affected_states = ['affected', 'nonaffected']
        for biomarker in shuffled_biomarkers:
            bm_params = params[biomarker]
            for state in affected_states:
                irreg_dict[biomarker][state] = very_irregular_distribution(
                    biomarker, bm_params, state=state, rng = rng)

    for participant_id, (k_j, is_diseased) in enumerate(zip(all_kjs, all_diseased)):
        for biomarker in shuffled_biomarkers:
            bm_params = params[biomarker]
            event_time = event_time_dict[biomarker]
            
            # Determine measurement state
            if is_diseased and k_j >= event_time:
                state = "affected"
                greek_letter = 'theta'
            else:
                state = "nonaffected"
                greek_letter = 'phi'

            # Generate Xnj based on experiment type
            if 'xnjNormal' in experiment_name:
                Xnj = rng.normal(bm_params[f"{greek_letter}_mean"], bm_params[f"{greek_letter}_std"])
            # Nonnormal distributions
            else:
                Xnj = rng.choice(irreg_dict[biomarker][state], size=1)[0]
            
            # Build data record
            record = {
                "participant": participant_id,
                "biomarker": biomarker,
                "measurement": Xnj,
                "diseased": is_diseased,
            }
            if keep_all_cols:
                record.update({
                    "event_time": event_time,
                    "k_j": k_j,
                    "affected": k_j >= event_time
                })
            data.append(record)
    return data

def generate_measurements_kjContinuous(
    event_time_dict, all_kjs, all_diseased, shuffled_biomarkers, params, keep_all_cols, rng = None):
    if rng is None:
        rng = np.random.default_rng()
    data = []
    for participant_id, (k_j, is_diseased) in enumerate(zip(all_kjs, all_diseased)):
        for n, biomarker in enumerate(shuffled_biomarkers):
            event_time = event_time_dict[biomarker]
            bm_params = params[biomarker]
            healthy_Xnj = rng.normal(bm_params["phi_mean"], bm_params["phi_std"])
            if is_diseased:
                # Diseased: sigmoid progression + noise
                R_i = bm_params["R_i"]
                rho_i = bm_params["rho_i"]
                sigmoid_term = R_i / (1 + np.exp(-rho_i * (k_j - event_time)))
                Xnj = sigmoid_term + healthy_Xnj
            else:
                # Healthy: pure noise (no disease effect)
                Xnj = healthy_Xnj
            # Build data record
            record = {
                "participant": participant_id,
                "biomarker": biomarker,
                "measurement": Xnj,
                "diseased": is_diseased,
            }
            if keep_all_cols:
                record.update({
                    "event_time": event_time,
                    "k_j": k_j,
                    "affected": k_j >= event_time
                })
            
            data.append(record)
    return data 

def generate_data(
    experiment_name: str,
    params_file: str,
    n_participants: int,
    healthy_ratio: float,
    output_dir: str,
    m: int,  # combstr_m
    seed: int,
    dirichlet_alpha: Optional[List[float]],
    beta_params: Optional[Dict],
    prefix: Optional[str],
    suffix: Optional[str],
    keep_all_cols: Optional[bool],
    bm_event_time_dicts: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """
    Simulate an Event-Based Model (EBM) for disease progression.

    Args:

    experiment_name (str): experiment name
    params_file (str): Directory of the params.json 
    js (List[int]): List of numbers of participants.
    rs (List[float]): List of healthy ratios.
    num_of_datasets_per_combination (int): Number of datasets to generate per combination.
    output_dir (str): Directory to save the generated datasets.
    seed (Optional[int]): Global seed for reproducibility. If None, a random seed is used.
    stage_distribution (Optional[str]), chooose from "continuous_uniform", "continuous_beta",
        "discrete_uniform", and "discrete_dirichlet_multinomial"
    dirichlet_alpha: Optional[Dict[str, Union[float, List]]] = {'uniform': 100, 'multinomial':[]},
    beta_params: Optional[Dict[str, float]]
    prefix (Optional[str]): Optional prefix of filename
    suffix (Optional[str]): Optional suffix of filename
    keep_all_cols (Optional[bool]): if False, drop ['k_j', 'event_time', 'affected']
    bm_event_time_dicts (Dict): The dicts to contain all filename: correct_order_dict for this experiment

    Returns:
    pd.DataFrame: A DataFrame with columns 'participant', "biomarker", 'measurement', 
        'diseased', with or without ['k_j', 'event_time', 'affected']
    """
    # Parameter validation
    assert n_participants > 0, "Number of participants must be greater than 0."
    assert 0 <= healthy_ratio <= 1, "Healthy ratio must be between 0 and 1."
    assert experiment_name in experiment_names, "Wrong experiment name!"

    rng = np.random.default_rng(seed)

    # Load parameters
    with open(params_file) as f:
        params = json.load(f)

    # Calculate max_stage from actual biomarker orders

    # Randomly shuffle first
    shuffled_biomarkers = rng.permutation(np.array(list(params.keys())))
    max_stage = len(shuffled_biomarkers)
    event_times = np.arange(1, max_stage+1)

    # Generate disease status
    n_healthy = int(n_participants * healthy_ratio)
    n_diseased = n_participants - n_healthy

    # ================================================================
    # Core generation logic 
    # ================================================================

    # EBM-Native Generative Model 

    if "kjOrdinal" in experiment_name:
        # Ordinal event time (Randomized)
        event_time_dict = dict(zip(shuffled_biomarkers, event_times))
        # Generate diseased kjs 
        if 'Uniform' in experiment_name:
            # alpha = 100
            if len(dirichlet_alpha['uniform']) != max_stage:
                dirichlet_alphas = [dirichlet_alpha['uniform'][0]] * max_stage
            stage_probs = rng.dirichlet(dirichlet_alphas)
        else:
            # fixed alpha values
            stage_probs = rng.dirichlet(dirichlet_alpha['multinomial'])
        counts = rng.multinomial(n_diseased, stage_probs)
        kjs = np.repeat(np.arange(1, max_stage + 1), counts)

        # healthy + diseased participants
        all_kjs = np.concatenate([np.zeros(n_healthy), kjs])
        all_diseased = all_kjs > 0 

        # Shuffle participants
        shuffle_idx = rng.permutation(n_participants)
        all_kjs = all_kjs[shuffle_idx]
        all_diseased = all_diseased[shuffle_idx]

        data = generate_measurements_kjOrdinal(
            params, event_time_dict, shuffled_biomarkers, experiment_name, all_kjs, 
            all_diseased, keep_all_cols, rng = rng)
    
    else:
        # Generate event times
        if experiment_name.startswith('xi'):
            event_time_raw = rng.beta(
                a = beta_params['near_normal']['alpha'], 
                b = beta_params['near_normal']['beta'], 
                size=max_stage)
            # Scale to [0, N]
            event_times = np.clip(event_time_raw * max_stage, 0, max_stage)
        
        event_time_dict = dict(zip(shuffled_biomarkers, event_times))

        # Generate kjs for diseased participants
        if 'kjContinuousUniform' in experiment_name:
            kjs_raw = rng.beta(
                a = beta_params['uniform']['alpha'],
                b = beta_params['uniform']['beta'],
                size = n_diseased
            )
        else:
            kjs_raw = rng.beta(
                a = beta_params['regular']['alpha'],
                b = beta_params['regular']['beta'],
                size = n_diseased
            )
        # Scale to [0, N]
        kjs = np.clip(kjs_raw * max_stage, 0, max_stage)

        # healthy + diseased participants
        all_kjs = np.concatenate([np.zeros(n_healthy), kjs])
        all_diseased = all_kjs > 0 

        # Shuffle participants
        shuffle_idx = rng.permutation(n_participants)
        all_kjs = all_kjs[shuffle_idx]
        all_diseased = all_diseased[shuffle_idx]

        data = generate_measurements_kjContinuous(
            event_time_dict, all_kjs, all_diseased, shuffled_biomarkers, params, keep_all_cols, rng = rng)

    # Save to CSV
    df = pd.DataFrame(data)
    filename = f"{int(healthy_ratio*n_participants)}_{n_participants}_{experiment_name}_{m}"
    filename = f"j{n_participants}_r{healthy_ratio}_E{experiment_name}_m{m}"
    if prefix: filename = f"{prefix}_{filename}"
    if suffix: filename = f"{filename}_{suffix}"
    df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)

    # In case bm_event_time_dicts has continuous event times, use the original version
    bm_event_time_dicts[filename] = dict(zip(
        sorted(event_time_dict, key=lambda x: event_time_dict[x]), 
        np.arange(1, max_stage + 1)))

    return df

def generate(
    experiment_name: str = "sn_kjOrdinalDM_xnjNormal",
    params_file: str = 'params.json',
    js: List[int] = [50, 200, 500, 1000],
    rs: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    num_of_datasets_per_combination: int = 50,
    output_dir: str = 'data',
    seed: Optional[int] = None,
    dirichlet_alpha: Optional[Dict[str, List]] = {'uniform': [100], 
    'multinomial':[0.4013728324975898,
                    1.0910444770153345,
                    2.30974117596663,
                    3.8081194066281103,
                    4.889722107892335,
                    4.889722107892335,
                    3.8081194066281103,
                    2.30974117596663,
                    1.0910444770153345,
                    0.4013728324975898]},
    beta_params: Dict[str, Dict[str, float]] = {
        'near_normal': {'alpha': 2.0, 'beta': 2.0},
        'uniform': {'alpha': 1, 'beta': 1},
        'regular': {'alpha': 5, 'beta': 2}
    },
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    keep_all_cols: Optional[bool] = False 
):
    """
    Generates datasets for multiple combinations of participants, healthy ratios, and datasets.

    Args:
    experiment_name (str): experiment name
    params_file (str): Directory of the params.json 
    js (List[int]): List of numbers of participants.
    rs (List[float]): List of healthy ratios.
    num_of_datasets_per_combination (int): Number of datasets to generate per combination.
    output_dir (str): Directory to save the generated datasets.
    seed (Optional[int]): Global seed for reproducibility. If None, a random seed is used.
    stage_distribution (Optional[str]), chooose from "continuous_uniform", "continuous_beta",
        "discrete_uniform", and "discrete_dirichlet_multinomial"
    dirichlet_alpha: Optional[Dict[str, Union[float, List]]] = {'uniform': 100, 'multinomial':[]},
    beta_params: Optional[Dict[str, float]]
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

    # biomarker_event_time_dict 
    # filename: {biomarker: event_time}
    bm_event_time_dicts = {}

    for j in js:
        for r in rs:
            for variant in range(num_of_datasets_per_combination):
                sub_seed = rng.integers(0, 1_000_000)
                generate_data(
                    experiment_name=experiment_name,
                    params_file=params_file,
                    n_participants=j,
                    healthy_ratio=r,
                    output_dir=output_dir,
                    m=variant,
                    seed=sub_seed,
                    dirichlet_alpha=dirichlet_alpha,
                    beta_params = beta_params,
                    prefix=prefix,
                    suffix=suffix,
                    keep_all_cols=keep_all_cols,
                    bm_event_time_dicts=bm_event_time_dicts,
                )
    print(f"Data generation complete. Files saved in {output_dir}/")
    return bm_event_time_dicts