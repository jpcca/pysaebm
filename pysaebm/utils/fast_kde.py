import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange, float64, int32
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd
import math 
from pysaebm.utils.kmeans import get_two_clusters_with_kmeans

# Constants
EPSILON = 1e-12
SQRT_2PI = math.sqrt(2.0 * math.pi)

@njit(fastmath=True)
def gaussian_kernel(x: float, data_point: float, bw: float) -> float:
    """ data point is one of the measurement in all measurements of a biomarker across all participants
    """
    z = (x - data_point) / bw
    return math.exp(-0.5 * z * z) / (bw * SQRT_2PI)

@njit(fastmath=True)
def calculate_bandwidth(data: np.ndarray, weights: np.ndarray, bw_method:str) -> float:
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
    if bw_method == "scott":
        return sigma * n ** (-0.2)
    elif bw_method == "silverman":
        return sigma * (4.0 / (3.0 * n)) ** 0.2

@njit(fastmath=True)
def _compute_pdf(x: float, data: np.ndarray, weights: np.ndarray, bw: float) -> float:
    pdf = 0.0 
    for j in range(len(data)):
        pdf += weights[j] * gaussian_kernel(x, data[j], bw)
    return pdf 

@njit(fastmath=True)
def _compute_ln_likelihood_kde_core(
    measurements: np.ndarray, 
    kde_data: np.ndarray,        
    kde_weights: np.ndarray,  
    bw_method: str
) -> float:
    """
    Compute KDE log PDF efficiently using Numba.
    
    Args:
        measurements: Biomarker measurements for a specific individual
        kde_data: KDE sample points
        kde_weights: KDE weights
        bw_method
        
    Returns:
        Total log PDF value
    """
    total = 0.0
    n = len(measurements)
    for i in range(n): # index of biomarker and also the corresponding measurement
        x = measurements[i]
        bm_data = kde_data[i]  # all the measurements for this bm across all participants
        weights = kde_weights[i]
        bw = calculate_bandwidth(bm_data, weights, bw_method)
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
    bw_method: str
) -> float:
    """
    Optimized KDE likelihood computation.
    
    Args:
        measurements: Biomarker measurements for a specific individual
        S_n: Stage thresholds
        biomarkers: Biomarker identifiers
        k_j: Stage value
        kde_dict: Dictionary of KDE objects for each biomarker,
        bw_method: method for bandwidth selection
        
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
        bw_method
    )

def get_initial_kde_estimates(
    data: pd.DataFrame
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
        
        # Get measurements as a 1D array
        measurements = biomarker_df['measurement'].to_numpy()
        
        # Get initial clusters using KMeans
        theta_measurements, phi_measurements, is_theta = get_two_clusters_with_kmeans(biomarker_df)
        
        # Normalize weights
        theta_weights = is_theta.astype(np.float64)
        theta_sum = np.sum(theta_weights)
        if theta_sum > 0:
            theta_weights = theta_weights / theta_sum
            
        phi_weights = (1 - is_theta).astype(np.float64)
        phi_sum = np.sum(phi_weights)
        if phi_sum > 0:
            phi_weights = phi_weights / phi_sum

        estimates[biomarker] = {
            'data': measurements,
            'theta_weights': theta_weights,
            'phi_weights': phi_weights,
        }
    return estimates

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
    biomarker: str,
    participants: np.ndarray,
    measurements: np.ndarray,
    diseased: np.ndarray,
    stage_post: Dict[int, np.ndarray],
    theta_phi_current: Dict[str, Dict[str, np.ndarray]],
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
    phi_weights = np.zeros_like(measurements, dtype=np.float64)

    # Get adaptive threshold based on data size
    weight_change_threshold = get_adaptive_weight_threshold(data_size)

    # Update weights based on current posterior estimates
    for i, (p, d) in enumerate(zip(participants, diseased)):
        if not d:
            # For non-diseased participants, all weight goes to phi
            phi_weights[i] = 1.0
            theta_weights[i] = 0.0
        else:
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
    current_theta_weights = theta_phi_current[biomarker]['theta_weights']
    current_phi_weights = theta_phi_current[biomarker]['phi_weights']
    
    # Only update KDEs if weights changed significantly
    if np.mean(np.abs(theta_weights - current_theta_weights)) < weight_change_threshold:
        theta_weights = current_theta_weights  # Reuse existing weights
    
    if np.mean(np.abs(phi_weights - current_phi_weights)) < weight_change_threshold:
        phi_weights = current_phi_weights  # Reuse existing weights

    return theta_weights, phi_weights