import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from numba import njit, prange, float64, int32
from typing import Dict, List, Tuple, Union, Optional, Any
import numba
from alabebm.utils.kmeans import get_two_clusters_with_kmeans

# Constants
EPSILON = 1e-10
SQRT_2PI = 0.3989422804014327  # 1/sqrt(2*pi)
CACHE_PRECISION = 6  # Decimal places for cache keys

@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def gaussian_kernel(x: float, xi: float, bandwidth: float) -> float:
    u = (x - xi) / bandwidth
    return SQRT_2PI * np.exp(-0.5 * u * u) / bandwidth

@njit(float64(float64, float64[:], float64[:], float64), parallel=True, cache=True)
def weighted_kde_evaluate(x: float, samples: np.ndarray, weights: np.ndarray, bandwidth: float) -> float:
    result = 0.0
    for i in prange(len(samples)):
        result += weights[i] * gaussian_kernel(x, samples[i], bandwidth)
    return result

@njit(parallel=True, cache=True)
def batch_weighted_kde_evaluate(points: np.ndarray, samples: np.ndarray, 
                               weights: np.ndarray, bandwidth: float) -> np.ndarray:
    results = np.zeros(len(points), dtype=np.float64)
    for i in prange(len(points)):
        results[i] = weighted_kde_evaluate(points[i], samples, weights, bandwidth)
    return results

class FastKDE:
    def __init__(self, data: np.ndarray, weights: Optional[np.ndarray] = None, 
                 bandwidth: Union[str, float] = 'silverman'):
        self.data = np.asarray(data, dtype=np.float64).flatten()
        
        if weights is None:
            self.weights = np.ones_like(self.data, dtype=np.float64)
        else:
            self.weights = np.asarray(weights, dtype=np.float64).flatten()
            
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:  # Handle all-zero weights
            self.weights = np.ones_like(self.weights) / len(self.weights)

        if bandwidth == 'silverman':
            n = len(self.data)
            sigma = max(np.std(self.data), 1e-6)
            self.bandwidth = sigma * (4 / (3 * n)) ** (1/5)
        else:
            self.bandwidth = max(float(bandwidth), 1e-6)
            
        self.tree = cKDTree(self.data[:, np.newaxis])
        self._cache = {}

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).flatten()
        
        # Use batch evaluation for all points with cache lookup
        results = np.zeros(len(points), dtype=np.float64)
        uncached_points = []
        uncached_indices = []
        
        # Check cache with rounded keys
        for i, x in enumerate(points):
            x_rounded = round(x, CACHE_PRECISION)
            if x_rounded in self._cache:
                results[i] = self._cache[x_rounded]
            else:
                uncached_points.append(x)
                uncached_indices.append(i)
        
        if uncached_points:
            # Find all relevant indices for uncached points
            all_indices = self.tree.query_ball_point(
                np.array(uncached_points)[:, np.newaxis], 
                3 * self.bandwidth
            )
            
            # Evaluate uncached points
            uncached_results = batch_weighted_kde_evaluate(
                np.array(uncached_points),
                self.data,
                self.weights,
                self.bandwidth
            )
            
            # Update results and cache
            for idx, res in zip(uncached_indices, uncached_results):
                results[idx] = res
                if len(self._cache) < 1000:
                    self._cache[round(points[idx], CACHE_PRECISION)] = res
        
        return results

    def logpdf(self, points: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(self.evaluate(points), EPSILON))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FastKDE):
            return False
        return (np.array_equal(self.data, other.data) and 
                np.array_equal(self.weights, other.weights) and
                self.bandwidth == other.bandwidth)

@njit(cache=True)
def _compute_kde_logpdf(
    measurements: np.ndarray, 
    stages: np.ndarray, 
    bio_indices: np.ndarray,
    kde_data: np.ndarray, 
    kde_weights: np.ndarray,
    kde_lengths: np.ndarray,  # New array for valid data lengths
    bandwidths: np.ndarray    # Now 2D array [biomarker, theta(0)/phi(1)]
) -> float:
    total = 0.0
    
    for i in range(len(measurements)):
        idx = bio_indices[i]
        x = measurements[i]
        stage = stages[i]
        
        data_row = idx*2 + (0 if stage else 1)
        bw = bandwidths[idx, 0 if stage else 1]
        data = kde_data[data_row]
        weights = kde_weights[data_row]
        length = kde_lengths[data_row]

        pdf = 0.0
        for j in range(length):
            pdf += weights[j] * gaussian_kernel(x, data[j], bw)
        
        total += np.log(max(pdf, EPSILON))
    
    return total

def compute_ln_likelihood_kde_fast(
    measurements: np.ndarray, 
    S_n: int, 
    biomarkers: np.ndarray, 
    k_j: np.ndarray, 
    kde_dict: Dict[str, Dict[str, Union[FastKDE, np.ndarray]]]
) -> float:
    stages = (k_j >= S_n).astype(np.int32)
    unique_bios = np.unique(biomarkers)
    bio_to_idx = {b: i for i, b in enumerate(unique_bios)}
    bio_indices = np.array([bio_to_idx[b] for b in biomarkers], dtype=np.int32)

    # Pre-allocate arrays with proper dimensions
    max_data_size = max(
        max(len(kde_dict[b]['theta_kde'].data), len(kde_dict[b]['phi_kde'].data))
        for b in unique_bios
    )
    
    kde_data = np.zeros((len(unique_bios)*2, max_data_size), dtype=np.float64)
    kde_weights = np.zeros((len(unique_bios)*2, max_data_size), dtype=np.float64)
    kde_lengths = np.zeros(len(unique_bios)*2, dtype=np.int32)
    bandwidths = np.zeros((len(unique_bios), 2), dtype=np.float64)

    for i, b in enumerate(unique_bios):
        theta_kde = kde_dict[b]['theta_kde']
        phi_kde = kde_dict[b]['phi_kde']
        
        # Theta KDE (stage 1)
        theta_len = len(theta_kde.data)
        kde_data[i*2, :theta_len] = theta_kde.data
        kde_weights[i*2, :theta_len] = theta_kde.weights
        kde_lengths[i*2] = theta_len
        bandwidths[i, 0] = theta_kde.bandwidth

        # Phi KDE (stage 0)
        phi_len = len(phi_kde.data)
        kde_data[i*2+1, :phi_len] = phi_kde.data
        kde_weights[i*2+1, :phi_len] = phi_kde.weights
        kde_lengths[i*2+1] = phi_len
        bandwidths[i, 1] = phi_kde.bandwidth

    return _compute_kde_logpdf(
        measurements.astype(np.float64),
        stages,
        bio_indices,
        kde_data,
        kde_weights,
        kde_lengths,
        bandwidths
    )

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
    theta_phi_current: Dict[str, Union[FastKDE, np.ndarray]],
    disease_stages: np.ndarray,
    curr_order: int,
) -> Tuple[FastKDE, np.ndarray, FastKDE, np.ndarray]:
    measurements = np.array(measurements, dtype=np.float64)
    data_size = len(measurements)
    theta_weights = np.zeros_like(measurements, dtype=np.float64)
    phi_weights = np.zeros_like(measurements, dtype=np.float64)

    for i, (p, d) in enumerate(zip(participants, diseased)):
        if not d:
            phi_weights[i] = 1.0
        else:
            probs = stage_post[p]
            theta_weights[i] = np.sum(probs[disease_stages >= curr_order])
            phi_weights[i] = np.sum(probs[disease_stages < curr_order])

    # Normalize with safety checks
    theta_weight_sum = np.sum(theta_weights)
    phi_weight_sum = np.sum(phi_weights)
    
    if theta_weight_sum > EPSILON:
        theta_weights /= theta_weight_sum
    else:
        theta_weights = np.ones_like(theta_weights) / len(theta_weights)
        
    if phi_weight_sum > EPSILON:
        phi_weights /= phi_weight_sum
    else:
        phi_weights = np.ones_like(phi_weights) / len(phi_weights)

    # Get adaptive threshold based on data size
    weight_change_threshold = get_adaptive_weight_threshold(data_size)
    
    # Update theta KDE only if weights changed significantly
    current_theta_weights = theta_phi_current[biomarker]['theta_weights']
    if np.mean(np.abs(theta_weights - current_theta_weights)) < weight_change_threshold:
        theta_kde = theta_phi_current[biomarker]['theta_kde']
    else:
        theta_kde = FastKDE(data=measurements, weights=theta_weights)
        theta_phi_current[biomarker]['theta_weights'] = theta_weights.copy()

    # Update phi KDE only if weights changed significantly
    current_phi_weights = theta_phi_current[biomarker]['phi_weights']
    if np.mean(np.abs(phi_weights - current_phi_weights)) < weight_change_threshold:
        phi_kde = theta_phi_current[biomarker]['phi_kde']
    else:
        phi_kde = FastKDE(data=measurements, weights=phi_weights)
        theta_phi_current[biomarker]['phi_weights'] = phi_weights.copy()

    return theta_kde, theta_weights, phi_kde, phi_weights

def get_initial_kde_estimates(
    data: pd.DataFrame
) -> Dict[str, Dict[str, Union[FastKDE, np.ndarray]]]:
    """
    Obtain initial KDE estimates for each biomarker.

    Args:
        data: DataFrame containing participant data with columns:
             - biomarker: Biomarker identifier
             - measurement: Biomarker measurement value

    Returns:
        Dictionary mapping biomarkers to their KDE parameters:
        {
            "biomarker1": {
                "theta_kde": FastKDE,
                "theta_weights": np.ndarray,
                "phi_kde": FastKDE,
                "phi_weights": np.ndarray,
            },
            ...
        }
    """
    estimates = {}
    biomarkers = data['biomarker'].unique()
    data_size = len(data.participant.unique())
    
    for biomarker in biomarkers:
        biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)

        # Get initial clusters
        theta_measurements, phi_measurements = get_two_clusters_with_kmeans(biomarker_df)
        
        # Create KDEs with uniform weights
        estimates[biomarker] = {
            'theta_kde': FastKDE(data=theta_measurements),
            'theta_weights': np.ones(data_size)/data_size,
            'phi_kde': FastKDE(data=phi_measurements),
            'phi_weights': np.ones(data_size)/data_size
        }
    
    return estimates