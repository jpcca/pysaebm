from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.stats import mode
from numba import njit
import sys 
from collections import defaultdict

def compute_initial_theta_phi_for_biomarker(biomarker_df, max_attempt = 100, seed = None):
    """get initial theta and phi parameters for this biomarker using seeded k-means (semi-supervised KMeans)
    input: 
        - biomarker_df: a pd.dataframe of a specific biomarker
    output: 
        - a tuple: theta_mean, theta_std, phi_mean, phi_std
    """
    if seed is not None:
        # Set the seed for numpy's random number generator
        rng = np.random.default_rng(seed)
    else:
        rng = np.random

    n_clusters = 2
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]

    # Initialize centroids
    healthy_seed = np.mean(measurements[healthy_df.index])
    diseased_seed = np.mean(measurements[np.setdiff1d(np.arange(len(measurements)), healthy_df.index)])
    init_centers = np.array([[healthy_seed], [diseased_seed]])

    curr_attempt = 0
    n_init_value = 50
    clustering_setup = KMeans(n_clusters=n_clusters, n_init=n_init_value, init = init_centers)
    
    while curr_attempt < max_attempt:
        clustering_result = clustering_setup.fit(measurements)
        predictions = clustering_result.labels_
        cluster_counts = np.bincount(predictions) # array([3, 2])
        
        # Exit if exactly two clusters and both have two or more elements
        if len(cluster_counts) == n_clusters and all(c > 1 for c in cluster_counts):
            break 
        curr_attempt += 1
    else:
        print(f"KMeans failed. Will go ahead and randomize the predictions.")
        predictions = rng.choice([0, 1], size=len(measurements))
        # After randomization, make sure healthy participants are in the same group. 
        predictions[healthy_df.index] = 0
        cluster_counts = np.bincount(predictions)
        # Check if two non-empty clusters exist:
        if len(cluster_counts) != n_clusters or not all(c > 1 for c in cluster_counts):
            raise ValueError(f"KMeans clustering failed to find valid clusters within max_attempt.")
    
    healthy_predictions = predictions[healthy_df.index]
    mode_result = mode(healthy_predictions, keepdims=False).mode
    phi_cluster_idx = mode_result[0] if isinstance(mode_result, np.ndarray) else mode_result
    theta_cluster_idx = 1 - phi_cluster_idx

    # Empty clusters to strore measurements
    clustered_measurements = [[] for _ in range(n_clusters)]
    # Store measurements into their cluster
    for i, prediction in enumerate(predictions):
        clustered_measurements[prediction].append(measurements[i][0])
    
    # Calculate means and standard deviations
    # Use ddof=1 for unbiased sample standard deviation
    theta_mean, theta_std = np.mean(
        clustered_measurements[theta_cluster_idx]), np.std(
            clustered_measurements[theta_cluster_idx], ddof=1)
    phi_mean, phi_std = np.mean(
        clustered_measurements[phi_cluster_idx]), np.std(
            clustered_measurements[phi_cluster_idx], ddof=1)
    
    if any(np.isnan(v) for v in [theta_std, phi_std, theta_mean, phi_mean]):
        raise ValueError("Invalid value (NaN) in estimates.")

    return theta_mean, theta_std, phi_mean, phi_std

def get_initial_theta_phi_estimates(
    data: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Obtain theta and phi estimates (mean and standard deviation) for each biomarker.

    Args:
    data (pd.DataFrame): DataFrame containing participant data with columns 'participant', 
        'biomarker', 'measurement', and 'diseased'.

    Returns:
    Dict[str, Dict[str, float]]: A dictionary where each key is a biomarker name,
        and each value is another dictionary containing the means and standard deviations 
        for theta and phi of that biomarker, with keys 'theta_mean', 'theta_std', 'phi_mean', 
        and 'phi_std'.
    """
    # empty hashmap of dictionaries to store the estimates
    estimates = {}
    biomarkers = data.biomarker.unique()
    for biomarker in biomarkers:
        # Filter data for the current biomarker
        # reset_index is necessary here because we will use healthy_df.index later
        biomarker_df = data[data['biomarker']
                            == biomarker].reset_index(drop=True)
        theta_mean, theta_std, phi_mean, phi_std = compute_initial_theta_phi_for_biomarker(
            biomarker_df)
        estimates[biomarker] = {
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'phi_mean': phi_mean,
            'phi_std': phi_std
        }
    return estimates

def estimate_params_exact(
    m0: float, 
    n0: float, 
    s0_sq: float, 
    v0: float, 
    data: np.ndarray
) -> Tuple[float, float]:
    """
    Estimate posterior mean and standard deviation using conjugate priors for a Normal-Inverse Gamma model.

    Args:
        m0 (float): Prior estimate of the mean (μ).
        n0 (float): Strength of the prior belief in m0.
        s0_sq (float): Prior estimate of the variance (σ²).
        v0 (float): Prior degrees of freedom, influencing the certainty of s0_sq.
        data (np.ndarray): Observed data (measurements).

    Returns:
        Tuple[float, float]: Posterior mean (μ) and standard deviation (σ).
    """
    # Data summary
    sample_mean = np.mean(data)
    sample_size = len(data)
    sample_var = np.var(data, ddof=1)  # ddof=1 for unbiased estimator

    # Update hyperparameters for the Normal-Inverse Gamma posterior
    updated_m0 = (n0 * m0 + sample_size * sample_mean) / (n0 + sample_size)
    updated_n0 = n0 + sample_size
    updated_v0 = v0 + sample_size
    updated_s0_sq = (1 / updated_v0) * ((sample_size - 1) * sample_var + v0 * s0_sq +
                                        (n0 * sample_size / updated_n0) * (sample_mean - m0)**2)
    updated_alpha = updated_v0/2
    updated_beta = updated_v0*updated_s0_sq/2

    # Posterior estimates
    mu_posterior_mean = updated_m0
    sigma_squared_posterior_mean = updated_beta/updated_alpha

    mu_estimation = mu_posterior_mean
    std_estimation = np.sqrt(sigma_squared_posterior_mean)

    return mu_estimation, std_estimation

def compute_theta_phi_biomarker_em(
    participants: np.ndarray,
    measurements: np.ndarray,
    diseased: np.ndarray,
    stage_likelihoods_posteriors: Dict[int, np.ndarray],
    disease_stages: np.ndarray,
    curr_order: int,
    ) -> Tuple[float, float, float, float]:
    """ Obtain biomarker's parameters using soft kmeans
    """

    # Obtain two responsibilites
    resp_affected = [
        sum(stage_likelihoods_posteriors[p][disease_stages >= curr_order]) if is_diseased else 0.0
        for p, is_diseased in zip(participants, diseased)
    ]

    resp_affected = np.array(resp_affected)
    resp_nonaffected = 1 - resp_affected

    sum_affected = max(np.sum(resp_affected), 1e-9)
    sum_nonaffected = max(np.sum(resp_nonaffected), 1e-9)

    # Weighted average
    theta_mean = np.sum(resp_affected*measurements)/sum_affected
    phi_mean = np.sum(resp_nonaffected*measurements)/sum_nonaffected

    # Weighted STD
    theta_std = np.sqrt(np.sum(resp_affected*(measurements - theta_mean)**2) / sum_affected)
    phi_std = np.sqrt(np.sum(resp_nonaffected*(measurements - phi_mean)**2) / sum_nonaffected)
    return theta_mean, theta_std, phi_mean, phi_std

def update_theta_phi_estimates(
    biomarker_data: Dict[str, Tuple[int, np.ndarray, np.ndarray, bool]],
    theta_phi_current: Dict[str, Dict[str, float]], # Current state’s θ/φ
    stage_likelihoods_posteriors: Dict[int, np.ndarray],
    disease_stages:np.ndarray,
    algorithm: str,
    prior_n: float,    # Weak prior (not data-dependent)
    prior_v: float     # Weak prior (not data-dependent)
    ) -> Dict[str, Dict[str, float]]:
    """Update theta and phi params for all biomarkers.

    Args:
        - algorithm (str): either 'conjugate_prior' or 'mle'
    """
    updated_params = defaultdict(dict)
    for biomarker, (
        curr_order, measurements, participants, diseased) in biomarker_data.items():
        dic = {'biomarker': biomarker}
        theta_phi_current_biomarker = theta_phi_current[biomarker]
        if algorithm not in ['conjugate_priors', "mle", 'em']:
            raise ValueError('Algorithm should be chosen among conjugate_priors, em, and mle! Check your spelling!')
        if algorithm == 'em': 
            theta_mean, theta_std, phi_mean, phi_std = compute_theta_phi_biomarker_em(
            participants,
            measurements,
            diseased,
            stage_likelihoods_posteriors,
            disease_stages,
            curr_order,
            )
        else:
            affected_cluster, non_affected_cluster = obtain_affected_and_non_clusters(
                participants,
                measurements,
                diseased,
                stage_likelihoods_posteriors,
                disease_stages,
                curr_order,
            )
            if algorithm == 'conjugate_priors':
                theta_mean, theta_std, phi_mean, phi_std = compute_theta_phi_biomarker_conjugate_priors(
                    affected_cluster, non_affected_cluster, theta_phi_current_biomarker, prior_n, prior_v)
            else:
                theta_mean, theta_std, phi_mean, phi_std = compute_theta_phi_biomarker_mle(
                    affected_cluster, non_affected_cluster, theta_phi_current_biomarker)

        updated_params[biomarker] = {
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'phi_mean': phi_mean,
            'phi_std': phi_std,
        }
    return updated_params

def obtain_affected_and_non_clusters(
    participants: np.ndarray,
    measurements: np.ndarray,
    diseased: np.ndarray,
    stage_likelihoods_posteriors: Dict[int, np.ndarray],
    disease_stages: np.ndarray,
    curr_order: int,
    ) -> Tuple[List[float], List[float]]:

    """
    Obtain both the affected and non-affected clusters for a single biomarker.

    Args:
        participants (np.ndarray): Array of participant IDs.
        measurements (np.ndarray): Array of measurements for the biomarker.
        diseased (np.ndarray): Boolean array indicating whether each participant is diseased.
        stage_likelihoods_posteriors (Dict[int, np.ndarray]): Dictionary mapping participant IDs to their stage likelihoods.
        disease_stages (np.ndarray): Array of stages considered diseased.
        curr_order (int): Current order of the biomarker.

    Returns:
        Tuple[float, float, float, float]: Mean and standard deviation for affected (theta) and non-affected (phi) clusters.
    """
    affected_cluster = []
    non_affected_cluster = []

    for idx, p in enumerate(participants):
        m = measurements[idx]
        if not diseased[idx]:
            non_affected_cluster.append(m)
        else:
            if curr_order == 1:
                affected_cluster.append(m)
            else:
                stage_likelihoods = stage_likelihoods_posteriors[p]
                affected_prob = np.sum(stage_likelihoods[disease_stages >= curr_order])
                non_affected_prob = np.sum(stage_likelihoods[disease_stages < curr_order])
                if affected_prob > non_affected_prob:
                    affected_cluster.append(m)
                elif affected_prob < non_affected_prob:
                    non_affected_cluster.append(m)
                else:
                    if np.random.random() > 0.5:
                        affected_cluster.append(m)
                    else:
                        non_affected_cluster.append(m)
    return affected_cluster, non_affected_cluster

def compute_theta_phi_biomarker_conjugate_priors(
    affected_cluster: List[float],
    non_affected_cluster: List[float],
    theta_phi_current_biomarker: Dict[str, float], # Current state’s θ/φ
    prior_n: float,
    prior_v: float
    ) -> Tuple[float, float, float, float]:
    """
    When data follows a normal distribution with unknown mean (μ) and unknown variance (σ²), 
    the normal-inverse gamma distribution serves as a conjugate prior for these parameters. 
    This means the posterior distribution will also be a normal-inverse gamma distribution after updating with observed data.

    Args:
        affected_cluster (List[float]): list of biomarker measurements
        non_affected_cluster (List[float]): list of biomarker measurements
        theta_phi_current_biomarker (Dict[str, float]): the current state's theta/phi for this biomarker
        prior_n (strength of belief in prior of mean), and prior_v (prior degree of freedom) are the weakly infomred priors. 

    Returns:
        Tuple[float, float, float, float]: Mean and standard deviation for affected (theta) and non-affected (phi) clusters.
    """
    # --- Affected Cluster (Theta) ---
    if len(affected_cluster) < 2:  # Fallback if cluster has 0 or 1 data points
        theta_mean = theta_phi_current_biomarker['theta_mean']
        theta_std = theta_phi_current_biomarker['theta_std']
    else:
        theta_mean, theta_std = estimate_params_exact(
            m0=theta_phi_current_biomarker['theta_mean'], 
            # m0=np.mean(affected_cluster),
            n0=prior_n, 
            # s0_sq = np.var(affected_cluster, ddof=1),
            s0_sq=theta_phi_current_biomarker['theta_std']**2, 
            v0=prior_v, 
            data=affected_cluster
        )
    
    # --- Non-Affected Cluster (Phi) ---
    if len(non_affected_cluster) < 2:  # Fallback if cluster has 0 or 1 data points
        phi_mean = theta_phi_current_biomarker['phi_mean']
        phi_std = theta_phi_current_biomarker['phi_std']
    else:
        phi_mean, phi_std = estimate_params_exact(
            m0=theta_phi_current_biomarker['phi_mean'], 
            # m0=np.mean(non_affected_cluster),
            n0=prior_n, 
            # s0_sq = np.var(non_affected_cluster, ddof=1),
            s0_sq=theta_phi_current_biomarker['phi_std']**2, 
            v0=prior_v, 
            data=non_affected_cluster
        )
    return theta_mean, theta_std, phi_mean, phi_std

def compute_theta_phi_biomarker_mle(
    affected_cluster: List[float],
    non_affected_cluster: List[float],
    theta_phi_current_biomarker: Dict[str, float], # Current state’s θ/φ
    ) -> Tuple[float, float, float, float]:
    """
    maximum likelihood estimation (MLE)
    Treats parameters as fixed, unknown constants to be estimated.
    Relies only on observed data to compute estimates, ignoring prior information.

    Args:
        affected_cluster (List[float]): list of biomarker measurements
        non_affected_cluster (List[float]): list of biomarker measurements

    Returns:
        Tuple[float, float, float, float]: Mean and standard deviation for affected (theta) and non-affected (phi) clusters.
    """
    
    # Compute means and standard deviations
    theta_mean = np.mean(affected_cluster) if affected_cluster else theta_phi_current_biomarker['theta_mean']
    theta_std = np.std(affected_cluster, ddof=1) if len(affected_cluster) >= 2 else theta_phi_current_biomarker['theta_std']
    phi_mean = np.mean(
        non_affected_cluster) if non_affected_cluster else theta_phi_current_biomarker['phi_mean']
    phi_std = np.std(non_affected_cluster, ddof=1) if len(non_affected_cluster) >=2 else theta_phi_current_biomarker['phi_std']
    return theta_mean, theta_std, phi_mean, phi_std

def preprocess_participant_data(
    data_we_have: pd.DataFrame, current_order_dict: Dict
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Preprocess participant data into NumPy arrays for efficient computation.

    Args:
        data (pd.DataFrame): Raw participant data.
        current_order_dict (Dict): Mapping of biomarkers to stages.

    Returns:
        Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, bool]]: A dictionary where keys are participant IDs,
            and values are tuples of (measurements, S_n, biomarkers).
    """
    # Change the column of S_n inplace
    data_we_have = data_we_have.copy()
    data_we_have.loc[:, 'S_n'] = data_we_have['biomarker'].map(current_order_dict)

    participant_data = {}
    for participant, pdata in data_we_have.groupby('participant'):
        # Will be a numpy array
        measurements = pdata['measurement'].values 
        S_n = pdata['S_n'].values 
        biomarkers = pdata['biomarker'].values  
        participant_data[participant] = (measurements, S_n, biomarkers)
    return participant_data

def preprocess_biomarker_data(
    data_we_have: pd.DataFrame,
    current_order_dict: Dict,
    ) -> Dict[str, Tuple[int, np.ndarray, np.ndarray, bool]]:
    """
    Preprocess data into NumPy arrays for efficient computation.

    Args:
        data_we_have (pd.DataFrame): Raw participant data.

    Returns:
        Dict[str, Tuple[int, np.ndarray, np.ndarray, bool]]: A dictionary where keys are biomarker names,
            and values are tuples of (curr_order, measurements, participants, diseased).
    """
    # Change the column of S_n inplace
    # Ensuring that we are explicitly modifying data_we_have and not an ambiguous copy.
    data_we_have = data_we_have.copy()
    data_we_have.loc[:, 'S_n'] = data_we_have['biomarker'].map(current_order_dict)

    biomarker_data = {}
    for biomarker, bdata in data_we_have.groupby('biomarker'):
        # Sort by participant to ensure consistent ordering
        bdata = bdata.sort_values(by = 'participant', ascending = True)

        curr_order = current_order_dict[biomarker]
        measurements = bdata['measurement'].values 
        participants = bdata['participant'].values  
        diseased = bdata['diseased'].values
        biomarker_data[biomarker] = (curr_order, measurements, participants, diseased)
    return biomarker_data

def compute_total_ln_likelihood_and_stage_likelihoods(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict[str, float]],
    current_pi: np.ndarray,
    disease_stages: np.ndarray
    ) -> Tuple[float, Dict[int, np.ndarray]]:
    """Calculate the total log likelihood across all participants 
        and obtain stage_likelihoods_posteriors
    """
    total_ln_likelihood = 0.0 
    # This is only for diseased participants
    stage_likelihoods_posteriors = {}
    # num_disease_stages = len(disease_stages)

    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        if participant in non_diseased_ids:
            # Non-diseased participant (fixed k=0)
            ln_likelihood = compute_ln_likelihood(
                measurements, S_n, biomarkers, k_j = 0, theta_phi = theta_phi)
        else:
            # Diseased participant (sum over possible stages)
            ln_stage_likelihoods = np.array([
                compute_ln_likelihood(
                    measurements, S_n, biomarkers, k_j = k_j, theta_phi=theta_phi
                ) + np.log(current_pi[k_j-1])
                for k_j in disease_stages
            ])
            # Use log-sum-exp trick for numerical stability
            max_ln_likelihood = np.max(ln_stage_likelihoods)
            stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
            likelihood_sum = np.sum(stage_likelihoods)
            # Proof: https://hongtaoh.com/en/2024/12/14/log-sum-exp/
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)

            # if likelihood_sum == 0:
            #     # Edge case: All stages have effectively zero likelihood
            #     normalized_probs = np.ones(num_disease_stages) / num_disease_stages
            #     ln_likelihood = np.log(sys.float_info.min)
            # else:
            # Normalize probabilities and compute marginal likelihood
            # Proof:
            # exp(ln(a₁) - M) = exp(ln(a₁)) * exp(-M) = a₁ * exp(-M)
            # exp(ln(a₂) - M) = a₂ * exp(-M)
            # exp(ln(a₃) - M) = a₃ * exp(-M)
            # normalized_prob₁ = (a₁ * exp(-M)) / (a₁ * exp(-M) + a₂ * exp(-M) + a₃ * exp(-M))
            # = (a₁ * exp(-M)) / ((a₁ + a₂ + a₃) * exp(-M))
            # = a₁ / (a₁ + a₂ + a₃)
            stage_likelihoods_posteriors[participant] = stage_likelihoods/likelihood_sum

        total_ln_likelihood += ln_likelihood
    return total_ln_likelihood, stage_likelihoods_posteriors

@njit
def _compute_ln_likelihood_core(measurements, mus, stds):
    """Core computation function optimized with Numba"""
    ln_likelihood = 0.0
    log_two_pi = np.log(2 * np.pi)
    two_times_pi = 2 * np.pi
    for i in range(len(measurements)):
        var = stds[i] ** 2
        diff = measurements[i] - mus[i]
        # likelihood *= np.exp(-diff**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
        # Log of normal PDF: ln(1/sqrt(2π*var) * exp(-diff²/2var))
        # = -ln(sqrt(2π*var)) - diff²/2var
        ln_likelihood += (-0.5 * (log_two_pi + np.log(var)) - diff**2 / (2 * var))
    return ln_likelihood

def compute_ln_likelihood(
    measurements: np.ndarray,
    S_n: np.ndarray,
    biomarkers: np.ndarray,
    k_j: int,
    theta_phi: Dict[str, Dict[str, float]]
) -> float:
    """
    Compute the log likelihood for given participant data.

    Args:
        measurements (np.ndarray): Array of measurement values.
        S_n (np.ndarray): Array of stage values (mapped from biomarkers).
        biomarkers (np.ndarray): Array of biomarker names.
        k_j (int): Current stage.
        theta_phi (Dict[str, Dict[str, float]]): Biomarker parameter dictionary.

    Returns:
        float: Log likelihood value.
    """
    mus = np.zeros(len(measurements))
    stds = np.zeros(len(measurements))
    affected = k_j >= S_n

    for i, (biomarker, is_affected) in enumerate(zip(biomarkers, affected)):
        params = theta_phi[biomarker]
        if is_affected:
            mus[i] = params['theta_mean']
            stds[i] = params['theta_std']
        else:
            mus[i] = params['phi_mean']
            stds[i] = params['phi_std']
    
    # Apply mask after mus and stds are computed
    valid_mask = (~np.isnan(measurements)) & (~np.isnan(mus)) & (stds > 0)
    measurements = measurements[valid_mask]
    mus = mus[valid_mask]
    stds = stds[valid_mask]

    return _compute_ln_likelihood_core(measurements, mus, stds)

def shuffle_order(arr: np.ndarray, n_shuffle: int) -> None:

    """
    Randomly shuffle a specified number of elements in an array.

    Args:
    arr (np.ndarray): The array to shuffle elements in.
    n_shuffle (int): The number of elements to shuffle within the array.
    """
    # Validate input 
    if n_shuffle <= 1:
        raise ValueError("n_shuffle must be >= 2 or =0")
    if n_shuffle > len(arr):
        raise ValueError("n_shuffle cannot exceed array length")
    if n_shuffle == 0:
        return 

    # Select indices and extract elements
    indices = np.random.choice(len(arr), size=n_shuffle, replace=False)
    original_indices = indices.copy()
    
    while True:
        shuffled_indices = np.random.permutation(original_indices)
        # Full derangement: make sure no indice stays in its original place
        if not np.any(shuffled_indices == original_indices):
            break 
    arr[indices] = arr[shuffled_indices]

def obtain_most_likely_order_dic(all_current_accepted_order_dicts, burn_in, thining):
    """Obtain the most likely order based on all the accepted orders 
    Inputs:
        - all_current_accepted_order_dicts 
        - burn_in
        - thining
    Outputs:
        - a dictionary where key is biomarker and value is the most likely order for that biomarker
    """
    biomarker_stage_probability_df = get_biomarker_stage_probability(
        all_current_accepted_order_dicts, burn_in, thining)
    dic = {}
    assigned_stages = set()

    # Prioritize biomarkers with the highest maximum stage probability
    sorted_biomarkers = sorted(
        biomarker_stage_probability_df.index,
        key=lambda x: max(biomarker_stage_probability_df.loc[x]),
        reverse=True # Sort descending by highest probability
    )

    for biomarker in sorted_biomarkers:
        # Get probability array for this biomarker
        # The first number will be the prob of this biomarker in stage 1, etc. 
        prob_arr = np.array(biomarker_stage_probability_df.loc[biomarker])

        # Sort indices of probabilities in descending order
        sorted_indices = np.argsort(prob_arr)[::-1] + 1 # Stages are 1-based

        for stage in sorted_indices:
            if stage not in assigned_stages:
                dic[biomarker] = int(stage)
                assigned_stages.add(stage)
                break
        else:
            raise ValueError(
                f"Could not assign a unique stage for biomarker {biomarker}.")
    return dic

def get_biomarker_stage_probability(all_current_accepted_order_dicts, burn_in, thining):
    """filter through all_dicts using burn_in and thining 
    and for each biomarker, get probability of being in each possible stage

    Input:
        - all_current_accepted_order_dicts 
        - burn_in
        - thinning
    Output:
        - dff: a pandas dataframe where index is biomarker name, each col is each stage
        and each cell is the probability of that biomarker indicating that stage

        Note that in dff, its index follows the same order as data_we_have.biomarker.unique()

        Also note that it is guaranteed that the cols will be corresponding to state 1, 2, 3, ... in an asending order
    """
    df = pd.DataFrame(all_current_accepted_order_dicts)
    df = df[(df.index > burn_in) & (df.index % thining == 0)]
    # Create an empty list to hold dictionaries
    dict_list = []

    # biomarkers are in the same order as data_we_have.biomarker.unique()
    biomarkers = np.array(df.columns)

    # iterate through biomarkers
    for biomarker in biomarkers:
        dic = {"biomarker": biomarker}
        # get the frequency of biomarkers
        # value_counts will generate a Series where index is each cell's value
        # and the value is the frequency of that value
        stage_counts = df[biomarker].value_counts()
        # for each stage
        # note that df.shape[1] should be equal to num_biomarkers
        for i in range(1, df.shape[1] + 1):
            # get stage:prabability
            dic[i] = stage_counts.get(i, 0)/len(df)
        dict_list.append(dic)

    dff = pd.DataFrame(dict_list)
    dff.set_index(dff.columns[0], inplace=True)
    return dff