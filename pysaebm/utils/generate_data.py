from typing import List, Optional, Tuple, Dict, Any
from . import data_processing as data_utils
import json 
import pandas as pd 
import numpy as np 
import os 
from collections import defaultdict, Counter
from bisect import bisect_right

# Rank continuous kjs
def get_rank(sorted_et:np.ndarray, val:float):
    # e.g., sorted_et = [0.1, 1.2, 3]
    # if val = 0 -> idx = 0; val = 0.3 -> idx = 2; val = 1.2 -> idx = 2
    idx = int(bisect_right(sorted_et, val))
    return idx  

def very_irregular_distribution(
    biomarker: str, 
    bm_params: Dict[str, float], 
    state: str = "affected", 
    size: int = 100000, 
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate highly irregular, non-normal samples for a given biomarker and state.
    
    This function creates complex, multi-modal distributions by combining different 
    probability distributions and transformations. Each biomarker receives a 
    custom-designed irregular distribution to simulate realistic biomarker behavior.
    
    Args:
        biomarker: The biomarker identifier (e.g., "MMSE", "AB", "HIP-FCI")
        bm_params: Dictionary containing distribution parameters (means and standard deviations)
        state: Whether the samples are for "affected" or "nonaffected" participants
        size: Number of samples to generate
        rng: Random number generator (if None, a new one will be created)
    
    Returns:
        np.ndarray: An array of irregularly distributed values with the requested size
    """
    if rng is None:
        rng = np.random.default_rng()

    # Select appropriate parameters based on disease state
    mean = bm_params["theta_mean"] if state == "affected" else bm_params["phi_mean"]
    std = bm_params["theta_std"] if state == "affected" else bm_params["phi_std"]

    # Initialize output array and split indices for multi-modal distributions
    base = np.zeros(size)
    segment_1, segment_2, segment_3 = np.array_split(np.arange(size), 3)

    # --- Highly non-normal design per biomarker ---
    if biomarker in ["MMSE", "ADAS"]:
        # Cognitive tests: Triangular + Normal + Exponential mixture
        base[segment_1] = rng.triangular(mean - 2*std, mean - 1.5*std, mean, size=len(segment_1))
        base[segment_2] = rng.normal(mean + std, 0.3 * std, size=len(segment_2))
        base[segment_3] = rng.exponential(scale=0.7 * std, size=len(segment_3)) + mean - 0.5 * std

    elif biomarker in ["AB", "P-Tau"]:
        # CSF biomarkers: Pareto + Uniform + Logistic mixture
        base[segment_1] = rng.pareto(1.5, size=len(segment_1)) * std + mean - 2 * std
        base[segment_2] = rng.uniform(mean - 1.5 * std, mean + 1.5 * std, size=len(segment_2))
        base[segment_3] = rng.logistic(loc=mean, scale=std, size=len(segment_3))

    elif biomarker in ["HIP-FCI", "HIP-GMI"]:
        # Hippocampus metrics: Beta + Exponential + Modified normal
        base[segment_1] = rng.beta(0.5, 0.5, size=len(segment_1)) * 4 * std + mean - 2 * std
        base[segment_2] = rng.exponential(scale=std * 0.4, size=len(segment_2)) * rng.choice([-1, 1], size=len(segment_2)) + mean
        base[segment_3] = rng.normal(mean, std * 0.5, size=len(segment_3)) + rng.choice([0, std * 2], size=len(segment_3))

    elif biomarker in ["AVLT-Sum", "PCC-FCI"]:
        # Memory and PCC metrics: Gamma + Weibull + Normal with spikes
        base[segment_1] = rng.gamma(shape=2, scale=0.5 * std, size=len(segment_1)) + mean - std
        base[segment_2] = rng.weibull(1.0, size=len(segment_2)) * std + mean - std
        base[segment_3] = rng.normal(mean, std * 0.5, size=len(segment_3)) + rng.choice([-1, 1], size=len(segment_3)) * std

    elif biomarker == "FUS-GMI":
        # Fusiform gyrus GMI: Heavy-tailed Cauchy with normal noise
        raw = rng.standard_cauchy(size=size) * std + mean
        raw += rng.normal(0, 0.2 * std, size=size)
        base = np.clip(raw, mean - 4 * std, mean + 4 * std)

    elif biomarker == "FUS-FCI":
        # Fusiform gyrus FCI: Bimodal with sharp spike
        spike_size = size // 10
        base[:spike_size] = rng.normal(mean, 0.2 * std, size=spike_size)
        base[spike_size:] = rng.logistic(loc=mean + std, scale=2 * std, size=size - spike_size)

    else:
        # Default for any other biomarker: Uniform distribution
        base = rng.uniform(mean - 2 * std, mean + 2 * std, size=size)

    # --- Add irregular noise and clip to reasonable range ---
    base += rng.normal(0, 0.2 * std, size=size)  # extra randomness
    base = np.clip(base, mean - 5 * std, mean + 5 * std)

    return base

def generate_measurements_kjOrdinal(
    params: Dict[str, Dict[str, float]], 
    event_time_dict: Dict[str, float], 
    shuffled_biomarkers: np.ndarray, 
    experiment_name: str, 
    all_kjs: np.ndarray, 
    all_diseased: np.ndarray, 
    keep_all_cols: bool, 
    rng: Optional[np.random.Generator] = None
) -> List[Dict[str, Any]]:
    """
    Generate measurements for the ordinal kj experiment types.
    
    This function creates biomarker measurements for each participant based on
    their disease status and disease stage (kj), following the ordinal stage model.
    
    Args:
        params: Dictionary of biomarker parameters
        event_time_dict: Dictionary mapping biomarkers to their event times
        shuffled_biomarkers: Array of biomarkers in randomized order
        experiment_name: Name of the experiment being run
        all_kjs: Array of disease stages for all participants
        all_diseased: Boolean array indicating disease status of participants
        keep_all_cols: Whether to include additional columns in output
        rng: Random number generator
        
    Returns:
        List of dictionaries containing measurement data for all participants and biomarkers
    """
    if rng is None:
        rng = np.random.default_rng()

    data = []
    # Pre-generate non-normal distributions for faster sampling if needed
    irreg_dict = defaultdict(dict)
    if "xnjNonNormal" in experiment_name:
        disease_states = ['affected', 'nonaffected']
        for biomarker in shuffled_biomarkers:
            bm_params = params[biomarker]
            for state in disease_states:
                irreg_dict[biomarker][state] = very_irregular_distribution(
                    biomarker, bm_params, state=state, rng=rng)

    for participant_id, (disease_stage, is_diseased) in enumerate(zip(all_kjs, all_diseased)):
        for biomarker in shuffled_biomarkers:
            bm_params = params[biomarker]
            event_time = event_time_dict[biomarker]
            
            # Determine measurement state based on disease stage vs event time
            if is_diseased and disease_stage >= event_time:
                state = "affected"
                distribution_param = 'theta'  # Parameter for affected state
            else:
                state = "nonaffected"
                distribution_param = 'phi'    # Parameter for non-affected state

            # Generate measurement based on experiment type
            if 'xnjNormal' in experiment_name:
                # Normal distribution sampling
                measurement = rng.normal(
                    bm_params[f"{distribution_param}_mean"], 
                    bm_params[f"{distribution_param}_std"]
                )
            else:
                # Non-normal distribution sampling from pre-generated distributions
                measurement = rng.choice(irreg_dict[biomarker][state], size=1)[0]
            
            # Build data record
            record = {
                "participant": participant_id,
                "biomarker": biomarker,
                "measurement": measurement,
                "diseased": is_diseased,
            }
            
            # Add additional columns if requested
            if keep_all_cols:
                record.update({
                    "event_time": event_time,
                    "k_j": disease_stage,
                    "affected": disease_stage >= event_time
                })
            data.append(record)
    return data

def generate_measurements_kjContinuous(
    experiment_name: str,
    event_time_dict: Dict[str, float],
    all_kjs: np.ndarray,
    all_diseased: np.ndarray,
    shuffled_biomarkers: np.ndarray,
    params: Dict[str, Dict[str, float]],
    keep_all_cols: bool,
    noise_std_parameter:float,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """
    Generate measurements for the continuous kj experiment types.
    
    This function creates biomarker measurements for each participant using a
    sigmoid progression model for diseased participants, based on their disease
    stage (continuous kj value).
    
    Args:
        experiment_name: Experiment identifier
        event_time_dict: Dictionary mapping biomarkers to their event times
        all_kjs: Array of disease stages for all participants (continuous values)
        all_diseased: Boolean array indicating disease status of participants
        shuffled_biomarkers: Array of biomarkers in randomized order
        params: Dictionary of biomarker parameters
        keep_all_cols: Whether to include additional columns in output
        rng: Random number generator
        
    Returns:
        List of dictionaries containing measurement data for all participants and biomarkers
    """
    if rng is None:
        rng = np.random.default_rng()
    
    data = []
    # 50% chance of flipping direction; precompute and store; making sure it's consistant across participants
    flip_directions = {b: (-1)**rng.binomial(1, 0.5) for b in shuffled_biomarkers}

    # Calculate R and rho During parameter loading (one-time):
    for biomarker in params:
        theta = params[biomarker]["theta_mean"]
        phi = params[biomarker]["phi_mean"]
        params[biomarker]["R"] = theta - phi  # Precompute
        
        theta_var = params[biomarker]["theta_std"] ** 2
        phi_var = params[biomarker]["phi_std"] ** 2
        params[biomarker]["rho"] = max(1, abs(theta - phi) / np.sqrt(theta_var + phi_var))
    for participant_id, (disease_stage, is_diseased) in enumerate(zip(all_kjs, all_diseased)):
        # If with noise, generate noises here
        max_stage = len(shuffled_biomarkers)
        if experiment_name.startswith('xiNearNormalWithNoise'):
            noise_std = np.sqrt(max_stage * noise_std_parameter)
            noises = rng.normal(0, noise_std, size=max_stage)
        for biomarker_idx, biomarker in enumerate(shuffled_biomarkers):
            event_time = event_time_dict[biomarker] 
            # If with noise, apply noise here
            if experiment_name.startswith('xiNearNormalWithNoise'):
                event_time = np.clip(event_time + noises[biomarker_idx], 0, max_stage)
            bm_params = params[biomarker]
            
            # Generate baseline healthy measurement
            healthy_measurement = rng.normal(bm_params["phi_mean"], bm_params["phi_std"])
            
            if is_diseased:
                # For diseased participants: apply sigmoid progression model
                # The further the disease stage is beyond the event time, the larger the effect
                # During measurement generation:
                progression_magnitude = flip_directions[biomarker] * bm_params["R"] # Maximum progression effect
                progression_rate = bm_params["rho"]     # Rate of progression
                
                # Sigmoid function for disease progression
                sigmoid_term = progression_magnitude / (1 + np.exp(-progression_rate * (disease_stage - event_time)))
                measurement = sigmoid_term + healthy_measurement
            else:
                # For healthy participants: just use the healthy baseline measurement
                measurement = healthy_measurement
                
            # Build data record
            record = {
                "participant": participant_id,
                "biomarker": biomarker,
                "measurement": measurement,
                "diseased": is_diseased,
            }
            
            # Add additional columns if requested
            if keep_all_cols:
                record.update({
                    "event_time": event_time,
                    "k_j": disease_stage,
                    "affected": disease_stage >= event_time
                })
            
            data.append(record)
    return data 

def generate_data(
        filename:str,
        experiment_name: str,
        params: Dict[str, Dict[str, float]],
        n_participants: int,
        healthy_ratio: float,
        output_dir: str,
        m: int,  # combstr_m
        seed: int,
        dirichlet_alpha: Dict[str, List[float]],
        beta_params: Dict[str, Dict[str, float]],
        prefix: Optional[str],
        suffix: Optional[str],
        keep_all_cols: bool,
        fixed_biomarker_order: bool,
        noise_std_parameter:float,
        true_order_and_stages_dict: Dict[str, Dict[str, int]],
    ) -> pd.DataFrame:
    """
    Simulate an Event-Based Model (EBM) for disease progression and generate a dataset.
    
    This function generates a dataset for a specific experiment configuration, 
    participant count, and healthy ratio. It handles the simulation of disease 
    stages, biomarker ordering, and creates measurements for each participant.

    Args:
        filename: the name for this data file to be saved. 
        experiment_name: Identifier for the experiment type
        params: Dictionary of biomarker parameters
        n_participants: Total number of participants to simulate
        healthy_ratio: Proportion of participants to be simulated as healthy
        output_dir: Directory to save the resulting CSV
        m: Dataset variant number (for generating multiple datasets with same parameters)
        seed: Random seed for reproducibility
        dirichlet_alpha: Parameters for Dirichlet distribution for stage distribution
        beta_params: Parameters for various Beta distributions used in the simulations
        prefix: Optional prefix for the output filename
        suffix: Optional suffix for the output filename
        keep_all_cols: Whether to include additional metadata columns in the output
        fixed_biomarker_order: If True, will use the order as in params_file. If False, will randomize the ordering.
        true_order_and_stages_dict: Dictionary to store the true biomarker ordering for evaluation
    
    Returns:
        pd.DataFrame: The generated dataset
    """
    # Parameter validation
    assert n_participants > 0, "Number of participants must be greater than 0."
    assert 0 <= healthy_ratio <= 1, "Healthy ratio must be between 0 and 1."

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Randomly shuffle biomarkers to create a ground truth ordering
    # This is correct. Biomarkers are shuffled and initial event_times are 1 to max_stage

    # If fixed biomarker order, I'll use the order as in the params
    if fixed_biomarker_order:
        shuffled_biomarkers = np.array(list(params.keys()))
    # Otherwise, randomize the order
    else:
        shuffled_biomarkers = rng.permutation(np.array(list(params.keys())))
    max_stage = len(shuffled_biomarkers)
    event_times = np.arange(1, max_stage+1)

    # Calculate counts of healthy and diseased participants
    n_healthy = int(n_participants * healthy_ratio)
    n_diseased = n_participants - n_healthy

    # ================================================================
    # Core generation logic based on experiment type
    # ================================================================

    if "kjOrdinal" in experiment_name:
        # Ordinal disease stage experiments (stages are discrete integers)
        
        # Assign event times to biomarkers in shuffled order
        event_time_dict = dict(zip(shuffled_biomarkers, event_times))
        
        # Generate stage distribution for diseased participants
        if 'Uniform' in experiment_name:
            # Use uniform Dirichlet prior (all stages equally likely)
            if len(dirichlet_alpha['uniform']) != max_stage:
                # If alpha parameter count doesn't match stages, replicate the first value
                dirichlet_alphas = [dirichlet_alpha['uniform'][0]] * max_stage
            else:
                dirichlet_alphas = dirichlet_alpha['uniform']
            stage_probs = rng.dirichlet(dirichlet_alphas)
            # # Integer division
            # stage_counts = [n_diseased // max_stage] * max_stage
            # remainder = n_diseased % max_stage
            # for i in range(remainder):
            #     stage_counts[i] += 1  # distribute the leftover
        else:
            # Use multinomial Dirichlet prior (custom stage distribution)
            stage_probs = rng.dirichlet(dirichlet_alpha['multinomial'][:max_stage])

        # Sample from multinomial to get actual stage counts
        stage_counts = rng.multinomial(n_diseased, stage_probs)
        
        # Create array of stages (1 to max_stage) for diseased participants
        disease_stages = np.repeat(np.arange(1, max_stage + 1), stage_counts)
        
        # Combine with healthy participants (stage 0)
        all_kjs = np.concatenate([np.zeros(n_healthy), disease_stages])
        all_diseased = all_kjs > 0 

        # Shuffle participant order
        shuffle_idx = rng.permutation(n_participants)
        all_kjs = all_kjs[shuffle_idx]
        all_diseased = all_diseased[shuffle_idx]

        # Generate measurements for all participants and biomarkers
        data = generate_measurements_kjOrdinal(
            params, event_time_dict, shuffled_biomarkers, experiment_name, all_kjs, 
            all_diseased, keep_all_cols, rng=rng)

        # For oridinal kjs and Sn, just use the kjs directly
        true_stages = [int(x) for x in all_kjs]
    
    else:
        # Continuous disease stage experiments (stages are real numbers)
        
        # Generate continuous event times for biomarkers
        if experiment_name.startswith('xi'):
            # Use beta distribution for near-normal event times
            event_time_raw = rng.beta(
                a=beta_params['near_normal']['alpha'], 
                b=beta_params['near_normal']['beta'], 
                size=max_stage)
            # Scale to [0, max_stage]
            event_times = event_time_raw * max_stage
        
        # Assign event times to biomarkers
        event_time_dict = dict(zip(shuffled_biomarkers, event_times))

        # Generate continuous disease stages (kj) for diseased participants
        if 'kjContinuousUniform' in experiment_name:
            # Use uniform-like beta for disease stages
            disease_stages_raw = rng.beta(
                a=beta_params['uniform']['alpha'],
                b=beta_params['uniform']['beta'],
                size=n_diseased
            )
        else:
            # Use skewed beta for disease stages
            disease_stages_raw = rng.beta(
                a=beta_params['regular']['alpha'],
                b=beta_params['regular']['beta'],
                size=n_diseased
            )
            
        # Scale disease stages to (0, max_stage]
        epsilon = 1e-8  # a small value
        disease_stages = np.clip(disease_stages_raw * max_stage, epsilon, max_stage)

        # Combine with healthy participants (stage 0)
        all_kjs = np.concatenate([np.zeros(n_healthy), disease_stages])
        all_diseased = all_kjs > 0 

        # Shuffle participant order because right now healthy is the first, and the diseased
        shuffle_idx = rng.permutation(n_participants)
        all_kjs = all_kjs[shuffle_idx]
        all_diseased = all_diseased[shuffle_idx]

        # Generate measurements using continuous disease progression model
        data = generate_measurements_kjContinuous(
            experiment_name, event_time_dict, all_kjs, all_diseased, 
            shuffled_biomarkers, params, keep_all_cols, noise_std_parameter=noise_std_parameter, rng=rng)
        
        sorted_event_times = sorted(event_times)
        true_stages = [get_rank(sorted_event_times, x) for x in all_kjs]

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    # Write to CSV
    df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)

    # Store the ground truth biomarker ordering for evaluation
    # Sort biomarkers by their event time and assign ordinal, sequential indices (1-based)
    true_order_and_stages_dict[filename]['true_order'] = dict(zip(
        sorted(event_time_dict, key=lambda x: event_time_dict[x]), 
        range(1, max_stage + 1)))
    true_order_and_stages_dict[filename]['true_stages'] = true_stages.copy()

    return df

def get_partial_orders(
        params:Dict[str, Dict[str, float]], 
        low:int, 
        high:int, 
        rng: np.random.Generator
    ) -> Tuple[List[str], List[str]]:
    # randomly pick two integers
    numbers = rng.choice(np.arange(low, high), size=2, replace=False)
    int1, int2 = numbers[0], numbers[1]

    # all biomarkers 
    bms = list(params.keys())

    # randomly select int1 and int2 biomarkers then shuffle
    partial_order1 = rng.choice(bms, size=int1, replace=False)
    partial_order2 = rng.choice(bms, size=int2, replace=False)
    rng.shuffle(partial_order1)
    rng.shuffle(partial_order2)
    return list(partial_order1), list(partial_order2)

def get_combined_order(
        params:Dict[str, Dict[str, float]], 
        low:int, 
        high:int, 
        seed:int) -> Tuple[List[str], List[List[str]]]:
    rng = rng = np.random.default_rng(seed)

    # two partial orders
    partial_order1, partial_order2 = get_partial_orders(params, low, high, rng)

    # make sure the two partial orders are different
    while partial_order1 == partial_order2:
        partial_order1, partial_order2 = get_partial_orders(params, low, high, rng)

    # prepare the input
    ordering_array = [partial_order1, partial_order2]

    # get combined ordering
    mpebm_mcmc_sampler = data_utils.MCMC(ordering_array=ordering_array, rng=rng)
    combined_order = mpebm_mcmc_sampler.obtain_sample_ordering()

    return combined_order, ordering_array

def get_final_params(
        params:Dict[str, Dict[str, float]], 
        combined_ordering:List[str], 
        ordering_array:List[List[str]]
    ) -> Dict[str, Dict[str, float]]:
    final_params = {}
    flattend_ordering_array = [item for sublist in ordering_array for item in sublist]
    frequency_dict = dict(Counter(flattend_ordering_array))

    for bm in combined_ordering:
        final_params[bm] = params[bm].copy()
        if frequency_dict[bm] > 1:
            final_params[bm]['theta_mean'] /= 0.9 
            final_params[bm]['theta_std'] /= 1.2

    return final_params

def generate(
    mixed_pathology: bool=False,
    experiment_name: str = "sn_kjOrdinalDM_xnjNormal",
    params_file: str = 'params.json',
    low: int=3,
    high: int=10,
    js: List[int] = [50, 200, 500, 1000],
    rs: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    num_of_datasets_per_combination: int = 50,
    output_dir: str = 'data',
    seed: Optional[int] = None,
    dirichlet_alpha: Optional[Dict[str, List[float]]] = {
        'uniform': [100], 
        'multinomial':[0.4013728324975898,
                      1.0910444770153345,
                      2.30974117596663,
                      3.8081194066281103,
                      4.889722107892335,
                      4.889722107892335,
                      3.8081194066281103,
                      2.30974117596663,
                      1.0910444770153345,
                      0.4013728324975898]
    },
    beta_params: Dict[str, Dict[str, float]] = {
        'near_normal': {'alpha': 2.0, 'beta': 2.0},
        'uniform': {'alpha': 1, 'beta': 1},
        'regular': {'alpha': 5, 'beta': 2}
    },
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    keep_all_cols: bool = False ,
    fixed_biomarker_order: bool = False,
    noise_std_parameter: float = 0.05
) -> Dict[str, Dict[str, int]]:
    """
    Generate multiple datasets for different experimental configurations.
    
    This function creates a suite of datasets with varying parameters for comprehensive
    evaluation of Event-Based Models (EBMs). It generates datasets for all combinations
    of participant counts, healthy ratios, and variants specified.
    
    Args:
        experiment_name: Type of experiment to run (see experiment_names for valid options)
        params_file: Path to JSON file with biomarker parameters
        low: the lowest when generating two random intergers
        high: the highest when generating two random intergers
        js: List of participant counts to generate data for
        rs: List of healthy ratios to generate data for
        num_of_datasets_per_combination: Number of dataset variants to generate per parameter combination
        output_dir: Directory to save generated CSV files
        seed: Master random seed (if None, a random seed will be generated)
        dirichlet_alpha: Parameters for Dirichlet distributions:
            - 'uniform': For uniform stage distribution
            - 'multinomial': For non-uniform stage distribution
        beta_params: Parameters for Beta distributions:
            - 'near_normal': For event times close to normal distribution
            - 'uniform': For approximately uniform distribution
            - 'regular': For skewed distribution
        prefix: Optional prefix for all filenames
        suffix: Optional suffix for all filenames
        keep_all_cols: Whether to include additional metadata columns (k_j, event_time, affected)
        fixed_biomarker_order: If True, will use the order as in params_file. If False, will randomize the ordering. 
        noise_std_parameter: the parameter in N(0, N \cdot noise_std_parameter) in experiment 9
        
    Returns:
        Dict mapping filenames to dictionaries of biomarker event time orderings
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load biomarker parameters from file
    with open(params_file) as f:
        params = json.load(f)

    if len(params) > len(dirichlet_alpha['multinomial']):
        raise ValueError(f"Your dirichlet_alpha multinomial must have the length of {len(params)}, as indicated by params.")

    # Initialize master random number generator
    if seed is None:
        seed = np.random.SeedSequence().entropy 
    rng = np.random.default_rng(seed)

    # Dictionary to store the ground truth biomarker orderings
    # Structure: {filename: true_order: {biomarker: event_time_position}, true_stages: []}
    true_order_and_stages_dict = defaultdict(dict)

    # Generate datasets for all parameter combinations
    for participant_count in js:
        for healthy_ratio in rs:
            for variant in range(num_of_datasets_per_combination):
                # Generate a unique seed for each dataset
                sub_seed = rng.integers(0, 1_000_000)

                # Construct filename with parameters encoded
                filename = f"j{participant_count}_r{healthy_ratio}_E{experiment_name}_m{variant}"
                if prefix: 
                    filename = f"{prefix}_{filename}"
                if suffix: 
                    filename = f"{filename}_{suffix}"

                """The goal here is use codes to automatically the final orderings, partial orderings, and their distribution parameters

                My idea is this: we have the original `params.json` for the ten biomarkers. I can

                - randomly pick two integers between 2-7, 
                - randomly select int1 and int2 biomarkers from `params.json`. Then shuffle them and use them as partial orderings.
                - generate a combined ordering based on the two partial orderings
                - Define a formula to determin the theta parameters for overlapping biomarkers. 
                - Generate data based on the combined ordering and the parameters, using pysaebm package as usual. 
                - Remember to save the two partial orderings to the `true_order_and_stages.json`. 
                """
                if mixed_pathology:
                    combined_ordering, ordering_array = get_combined_order(params, low, high, seed=sub_seed)
                    final_params = get_final_params(params, combined_ordering, ordering_array)
                    true_order_and_stages_dict[filename]['ordering_array'] = ordering_array
                
                # Generate a single dataset with the current parameter combination
                generate_data(
                    filename=filename,
                    experiment_name=experiment_name,
                    params = final_params,
                    n_participants=participant_count,
                    healthy_ratio=healthy_ratio,
                    output_dir=output_dir,
                    m=variant,
                    seed=sub_seed,
                    dirichlet_alpha=dirichlet_alpha,
                    beta_params=beta_params,
                    prefix=prefix,
                    suffix=suffix,
                    keep_all_cols=keep_all_cols,
                    fixed_biomarker_order = fixed_biomarker_order,
                    noise_std_parameter = noise_std_parameter,
                    true_order_and_stages_dict=true_order_and_stages_dict,
                )
                
    print(f"Data generation complete. Files saved in {output_dir}/")
    return true_order_and_stages_dict