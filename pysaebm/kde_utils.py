import numpy as np
import pandas as pd 
from collections import defaultdict
from numba import njit
from typing import Dict, Tuple
import math 
from sklearn.cluster import KMeans
from scipy.stats import mode

# Constants
EPSILON = 1e-12
SQRT_2PI = math.sqrt(2.0 * math.pi)

def get_two_clusters_with_kmeans(
    biomarker_df:pd.DataFrame,
    rng:np.random.Generator, 
    max_attempt: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """get affected and nonaffected clusters for a biomarker using seeded k-means (semi-supervised KMeans)
    input: 
        - biomarker_df: a pd.dataframe of a specific biomarker
    output: 
        - A Tuple: two arrays containing the measurements of each cluster (affected, and nonaffected)
        # Note that it is guaranteed that both clusters have at least 2 elements; otherwise, the program will stop. 
    """
    n_clusters = 2
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]

    # Initialize centroids
    healthy_seed = np.mean(measurements[healthy_df.index])
    diseased_seed = np.mean(measurements[np.setdiff1d(np.arange(len(measurements)), healthy_df.index)])
    init_centers = np.array([[healthy_seed], [diseased_seed]])

    curr_attempt = 0
    clustering_setup = KMeans(n_clusters=n_clusters, init = init_centers, n_init=1, random_state=42)
    
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
        # Initialize all predictions to -1 (or any placeholder value)
        predictions = np.full(len(measurements), -1)

        # Set healthy participants to 0
        predictions[healthy_df.index] = 0

        # Get indices of non-healthy participants
        non_healthy_indices = np.where(predictions == -1)[0]

        # Keep trying until both clusters have at least 2 members
        for _ in range(max_attempt):  # try up to 100 times
            # Randomly assign 0 or 1 to non-healthy participants
            predictions[non_healthy_indices] = rng.choice([0, 1], size=len(non_healthy_indices))
            cluster_counts = np.bincount(predictions)

            # Check if two non-empty clusters exist:
            if len(cluster_counts) == n_clusters and all(c > 1 for c in cluster_counts):
                break
        else:
            raise ValueError(f"KMeans clustering failed to find valid clusters within max_attempt.")
    
    healthy_predictions = predictions[healthy_df.index]
    mode_result = mode(healthy_predictions, keepdims=False).mode
    phi_cluster_idx = mode_result[0] if isinstance(mode_result, np.ndarray) else mode_result
    theta_cluster_idx = 1 - phi_cluster_idx

    # Convert predictions to numpy array if not already
    predictions = np.array(predictions).flatten()

    # Select affected and nonaffected measurements based on cluster index
    theta_measurements = measurements[predictions == theta_cluster_idx].flatten()
    phi_measurements = measurements[predictions == phi_cluster_idx].flatten()

    is_theta = (predictions == theta_cluster_idx).astype(np.float64)

    assert len(is_theta) == len(measurements), "is_theta should have the same length as measurements"

    return theta_measurements, phi_measurements, np.array(is_theta)

@njit(fastmath=False)
def gaussian_kernel(x: float, data_point: float, bw: float) -> float:
    """ data point is one of the measurement in all measurements of a biomarker across all participants
    """
    z = (x - data_point) / bw
    return math.exp(-0.5 * z * z) / (bw * SQRT_2PI)

@njit(fastmath=False)
def calculate_bandwidth(data: np.ndarray, weights: np.ndarray) -> float:
    """
        - data is all the measurements for a biomarker
        - weights are either the phi or theta weights
    """
    n = len(data)
    if n <= 1:
        return 1.0
    if weights is None or weights.size == 0:
        sigma = max(np.std(data), EPSILON)
    else:
        w_sum = max(np.sum(weights), EPSILON) # lower bound of sigma
        w_mean = np.sum(weights * data) / w_sum
        var = 0.0
        w2_sum = 0.0
        for i in range(n):
            diff = data[i] - w_mean
            var += weights[i] * diff * diff
            w2_sum += weights[i] * weights[i]
        sigma = max(math.sqrt(var / w_sum), EPSILON) # lower bound of sigma
        n_eff = 1.0 / max(w2_sum, EPSILON)
        n = n_eff
    return sigma * n ** (-0.2)

@njit(fastmath=False)
def _compute_pdf(x: float, data: np.ndarray, weights: np.ndarray, bw: str) -> float:
    pdf = 0.0 
    for j in range(len(data)):
        pdf += weights[j] * gaussian_kernel(x, data[j], bw)
    return pdf 

@njit(fastmath=False)
def _compute_ln_likelihood_kde_core(
    measurements: np.ndarray, 
    kde_data: np.ndarray,        
    kde_weights: np.ndarray,  
) -> float:
    """
    Compute KDE log PDF efficiently using Numba.
    
    Args:
        measurements: Biomarker measurements for a specific individual
        kde_data: KDE sample points
        kde_weights: KDE weights
        
    Returns:
        Total log PDF value
    """
    total = 0.0
    n = len(measurements)
    for i in range(n): # index of biomarker and also the corresponding measurement
        x = measurements[i]
        bm_data = kde_data[i]  # all the measurements for this bm across all participants
        weights = kde_weights[i]
        bw = calculate_bandwidth(bm_data, weights)
        pdf = _compute_pdf(x, bm_data, weights, bw)
        # Handle numerical stability
        total += np.log(max(pdf, EPSILON))
    return total

def compute_ln_likelihood_kde_fast(
    measurements: np.ndarray, 
    S_n: np.ndarray, 
    biomarkers: np.ndarray, 
    k_j: int, 
    kde_dict: Dict[str, np.ndarray],
) -> float:
    """
    Optimized KDE likelihood computation.
    
    Args:
        measurements: Biomarker measurements for a specific individual
        S_n: Stage thresholds
        biomarkers: Biomarker identifiers
        k_j: Stage value
        kde_dict: Dictionary of KDE objects for each biomarker,
        
    Returns:
        Log likelihood value
    """
    # Convert to stage indicators (1 for affected, 0 for non-affected)
    affected_flags = k_j >= S_n

    max_data_size = max(len(kde_dict[b]['data']) for b in biomarkers)

    # Pre-allocate arrays for Numba
    kde_data = np.zeros((len(biomarkers), max_data_size), dtype=np.float64)
    kde_weights = np.zeros((len(biomarkers), max_data_size), dtype=np.float64)
    
    # Fill arrays with data
    for i, b in enumerate(biomarkers):
        kde_data[i] = kde_dict[b]['data']
        
        kde_weights[i] = kde_dict[b]['theta_weights'] if affected_flags[i] else kde_dict[b]['phi_weights']
    
    # Compute log likelihood
    return _compute_ln_likelihood_kde_core(
        measurements,
        kde_data,
        kde_weights,
    )

def get_initial_kde_estimates(
    data: pd.DataFrame,
    rng:np.random.Generator
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Obtain initial KDE estimates for each biomarker.

    Args:
        data: DataFrame containing participant data

    Returns:
        Dictionary mapping biomarkers to their KDE parameters
    """
    estimates = {}
    biomarkers = data['biomarker'].unique()
    
    for biomarker in biomarkers:
        biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
        # Find the row‐indices of healthy cases in biomarker_df
        healthy_row_idx = biomarker_df.index[ biomarker_df['diseased']==False]

        # Get measurements as a 1D array
        measurements = biomarker_df['measurement'].to_numpy()
        
        # Get initial clusters using KMeans
        _, _, is_theta = get_two_clusters_with_kmeans(biomarker_df, rng)
        
        # Normalize weights
        theta_weights = is_theta.astype(np.float64)
        # for non_diseased participants, their theta must be 0 
        theta_weights[healthy_row_idx] = 0.0
        theta_sum = np.sum(theta_weights)
        if theta_sum > 0:
            theta_weights = theta_weights / theta_sum
            
        phi_weights = (1 - is_theta).astype(np.float64)
        # for non_diseased participants, their phi must be 1 
        phi_weights[healthy_row_idx] = 1.0
        phi_sum = np.sum(phi_weights)
        if phi_sum > 0:
            phi_weights = phi_weights / phi_sum

        estimates[biomarker] = {
            'data': measurements,
            'theta_weights': theta_weights,
            'phi_weights': phi_weights,
        }
    return estimates

@njit(fastmath=False)
def get_adaptive_weight_threshold(data_size: int) -> float:
    """Data-size dependent threshold for EM updates"""
    if data_size >= 1000:
        return 0.005
    elif data_size >= 500:
        return 0.0075
    elif data_size >= 200:
        return 0.01
    elif data_size >= 50:
        return 0.015
    else:
        return 0.02  # For very small datasets

def update_kde_for_biomarker_em(
    participants: np.ndarray,
    measurements: np.ndarray,
    diseased: np.ndarray,
    stage_post: Dict[int, np.ndarray],
    theta_phi_current_biomarker: Dict[str, np.ndarray],
    disease_stages: np.ndarray,
    curr_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update KDE estimates for a biomarker using EM with adaptive thresholds.
    
    Args:
        biomarker: Biomarker identifier
        participants: Participant IDs
        measurements: All measurements for this biomarker across all participants
        diseased: Disease status for participants
        stage_post: Stage posteriors from EM
        theta_phi_current: Current KDE estimates
        disease_stages: Disease stage values
        curr_order: Current biomarker order
        
    Returns:
        Updated theta_kde and phi_kde objects
    """
    data_size = len(measurements)
    
    # Initialize weight arrays
    theta_weights = np.zeros_like(measurements, dtype=np.float64)
    phi_weights = np.ones_like(measurements, dtype=np.float64)

    # Get adaptive threshold based on data size
    weight_change_threshold = get_adaptive_weight_threshold(data_size)

    # we've made sure the default theta is 0 and default phi is 1
    # so there is no need to go through non diseased participants
    # because for non-diseased participants, all weight goes to phi

    # Update weights based on current posterior estimates
    for i, (p, d) in enumerate(zip(participants, diseased)):
        if d:
            # For diseased participants, distribute weights based on stage
            probs = stage_post[p]
            theta_weights[i] = np.sum(probs[disease_stages >= curr_order])
            phi_weights[i] = np.sum(probs[disease_stages < curr_order])

    # Normalize weights
    theta_sum = np.sum(theta_weights)
    if theta_sum > 0:
        theta_weights /= theta_sum
    else:
        # Handle edge case with no theta weights
        theta_weights = np.ones_like(theta_weights) / len(theta_weights)
        
    phi_sum = np.sum(phi_weights)
    if phi_sum > 0:
        phi_weights /= phi_sum
    else:
        # Handle edge case with no phi weights
        phi_weights = np.ones_like(phi_weights) / len(phi_weights)

    # Theta KDE decision - compare new weights with current KDE weights
    # Access weights directly from the KDE objects
    current_theta_weights = theta_phi_current_biomarker['theta_weights']
    current_phi_weights = theta_phi_current_biomarker['phi_weights']
    
    # Only update KDEs if weights changed significantly
    if np.mean(np.abs(theta_weights - current_theta_weights)) < weight_change_threshold:
        theta_weights = current_theta_weights  # Reuse existing weights
    
    if np.mean(np.abs(phi_weights - current_phi_weights)) < weight_change_threshold:
        phi_weights = current_phi_weights  # Reuse existing weights

    return theta_weights, phi_weights


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
    data_we_have.loc[:, 'S_n'] = data_we_have['biomarker'].map(
        current_order_dict)

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
    data_we_have.loc[:, 'S_n'] = data_we_have['biomarker'].map(
        current_order_dict)

    biomarker_data = {}
    for biomarker, bdata in data_we_have.groupby('biomarker'):
        # Sort by participant to ensure consistent ordering
        bdata = bdata.sort_values(by='participant', ascending=True)

        curr_order = current_order_dict[biomarker]
        measurements = bdata['measurement'].values
        participants = bdata['participant'].values
        diseased = bdata['diseased'].values
        biomarker_data[biomarker] = (
            curr_order, measurements, participants, diseased)
    return biomarker_data

def update_theta_phi_estimates(
    biomarker_data: Dict[str, Tuple[int, np.ndarray, np.ndarray, bool]],
    theta_phi_current: Dict[str, Dict[str, float]],  # Current state’s θ/φ
    stage_likelihoods_posteriors: Dict[int, np.ndarray],
    disease_stages: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Update theta and phi params for all biomarkers.
    """
    updated_params = defaultdict(dict)
    for biomarker, (
            curr_order, measurements, participants, diseased) in biomarker_data.items():
        theta_phi_current_biomarker = theta_phi_current[biomarker]
        
        theta_weights, phi_weights = update_kde_for_biomarker_em(
            participants,
            measurements,
            diseased,
            stage_likelihoods_posteriors,
            theta_phi_current_biomarker,
            disease_stages,
            curr_order
        )
        updated_params[biomarker] = {
            'data': measurements,
            'theta_weights': theta_weights,
            'phi_weights': phi_weights,
        }

    return updated_params


def compute_total_ln_likelihood_and_stage_likelihoods(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict],
    current_pi: np.ndarray,
    disease_stages: np.ndarray,
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
            ln_likelihood = compute_ln_likelihood_kde_fast(
                measurements, S_n, biomarkers, k_j=0, kde_dict=theta_phi)
        else:
            # Diseased participant (sum over possible stages)
            ln_stage_likelihoods = np.array([
                compute_ln_likelihood_kde_fast(
                    measurements, S_n, biomarkers, k_j=k_j, kde_dict=theta_phi) + np.log(current_pi[k_j-1])
                for k_j in disease_stages
            ])
            
            # Use log-sum-exp trick for numerical stability
            max_ln_likelihood = np.max(ln_stage_likelihoods)
            stage_likelihoods = np.exp(
                ln_stage_likelihoods - max_ln_likelihood)
            likelihood_sum = np.sum(stage_likelihoods)
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)

            stage_likelihoods_posteriors[participant] = stage_likelihoods / \
                likelihood_sum

        total_ln_likelihood += ln_likelihood
    return total_ln_likelihood, stage_likelihoods_posteriors


def compute_unbiased_stage_likelihoods_kde(
        participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        theta_phi: Dict[str, Dict],
        current_pi: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Obtain stage_likelihoods_posteriors while ignoring the diagnosis label or diseased or not.
    """
    stage_likelihoods_posteriors = {}
    eps = 1e-12  # guard against log(0)

    for participant, (measurements, S_n, biomarkers) in participant_data.items():

        ln_stage_likelihoods = np.array([
            compute_ln_likelihood_kde_fast(
                measurements, S_n, biomarkers, k_j=k_j, kde_dict=theta_phi
            ) + np.log(current_pi[k_j] if current_pi[k_j] > eps else eps)
            for k_j in range(0, len(theta_phi) + 1) # possible stages now starts from 0; but that's okay because
            # S_n always starts from 1
        ])
        # Use log-sum-exp trick for numerical stability
        max_ln_likelihood = np.max(ln_stage_likelihoods)
        stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
        likelihood_sum = np.sum(stage_likelihoods)

        stage_likelihoods_posteriors[participant] = stage_likelihoods/likelihood_sum

    return stage_likelihoods_posteriors

def stage_with_plugin_pi_em_kde(
    participant_data:pd.DataFrame,
    final_theta_phi: np.ndarray,
    rng:np.random.Generator,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    """
    EM-calibrates full stage prior π over 0..N, then returns posteriors and MAP stages.
    Uses your compute_unbiased_stage_likelihoods internally. MH is untouched.

    sample inside MH, posterior mean outside MH.
    """
    n_participants = len(participant_data)
    n_stages = len(final_theta_phi) + 1 

    alpha_prior = np.ones(n_stages)
    pi = rng.dirichlet(alpha_prior)

    stage_post = None
    stage_post_matrix = np.zeros((n_participants, n_stages), dtype=np.float64)
    for _ in range(max_iter):
        # E-step: posteriors with current π (your existing function)
        stage_post = compute_unbiased_stage_likelihoods_kde(
            participant_data=participant_data,
            theta_phi=final_theta_phi,
            current_pi=pi,
        )

        for p, data in stage_post.items():
            stage_post_matrix[p] = data 

        # M-step (Dirichlet-MAP): counts + (alpha-1), then normalize
        counts = stage_post_matrix.sum(axis=0)
        pi_new = (alpha_prior + counts) / (alpha_prior.sum() + counts.sum())

        # Converged?
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            pi = pi_new
            break
        pi = pi_new

    # MAP stages (deterministic)
    ml_stages = np.argmax(stage_post_matrix, axis=1).astype(int)
    return stage_post_matrix, ml_stages, pi
