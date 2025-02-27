from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.stats import mode
from numba import njit
import sys 

def compute_theta_phi_for_biomarker(biomarker_df, max_attempt = 100, seed = None):
    """get theta and phi parameters for this biomarker using hard k-means
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

    curr_attempt = 0
    n_init_value = 50
    clustering_setup = KMeans(n_clusters=n_clusters, n_init=n_init_value)
    
    while curr_attempt < max_attempt:
        clustering_result = clustering_setup.fit(measurements)
        predictions = clustering_result.labels_
        cluster_counts = np.bincount(predictions) # array([3, 2])
        
        # Exit if exactly two clusters and neither one is empty
        if len(cluster_counts) == n_clusters and all(c > 1 for c in cluster_counts):
            break 
        curr_attempt += 1
    else:
        print(f"KMeans failed. Try randomizing the predictions")
        predictions = rng.choice([0, 1], size=len(measurements))
        cluster_counts = np.bincount(predictions)
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
    theta_mean, theta_std = np.mean(
        clustered_measurements[theta_cluster_idx]), np.std(
            clustered_measurements[theta_cluster_idx])
    phi_mean, phi_std = np.mean(
        clustered_measurements[phi_cluster_idx]), np.std(
            clustered_measurements[phi_cluster_idx])
    
    # Check for invalid values
    if any(np.isnan(v) or v == 0 for v in [theta_std, phi_std, theta_mean, phi_mean]):
        raise ValueError("One of the calculated values is invalid (0 or NaN).")

    return theta_mean, theta_std, phi_mean, phi_std

def get_theta_phi_estimates(
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
        theta_mean, theta_std, phi_mean, phi_std = compute_theta_phi_for_biomarker(
            biomarker_df)
        estimates[biomarker] = {
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'phi_mean': phi_mean,
            'phi_std': phi_std
        }
    return estimates

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

    for i, biomarker in enumerate(biomarker_stage_probability_df.index):
        # probability array for that biomarker
        prob_arr = np.array(biomarker_stage_probability_df.iloc[i, :])

        # Sort indices of probabilities in descending order
        sorted_indices = np.argsort(prob_arr)[::-1] + 1

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