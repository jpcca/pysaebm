import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange, float64, int32
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd
from sa_ebm.utils.kmeans import get_two_clusters_with_kmeans

# Constants
EPSILON = 1e-10
SQRT_2PI = 0.3989422804014327  # 1/sqrt(2*pi)
CACHE_PRECISION = 6  # Decimal places for cache keys

@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def gaussian_kernel(x: float, xi: float, bandwidth: float) -> float:
    """
    Gaussian kernel function optimized with Numba.
    
    Args:
        x: Point at which to evaluate the kernel
        xi: Sample point
        bandwidth: Kernel bandwidth parameter
        
    Returns:
        Kernel value at point x
    """
    u = (x - xi) / bandwidth
    return SQRT_2PI * np.exp(-0.5 * u * u) / bandwidth

@njit(float64(float64, float64[:], float64[:], float64), parallel=True, cache=True)
def weighted_kde_evaluate(x: float, samples: np.ndarray, weights: np.ndarray, bandwidth: float) -> float:
    """
    Evaluate KDE at point x using weighted samples with parallel processing.
    
    Args:
        x: Point at which to evaluate the KDE
        samples: Array of sample points
        weights: Array of weights for each sample
        bandwidth: Kernel bandwidth parameter
        
    Returns:
        Weighted KDE value at point x
    """
    result = 0.0
    for i in prange(len(samples)):
        result += weights[i] * gaussian_kernel(x, samples[i], bandwidth)
    return result

@njit(parallel=True, cache=True)
def batch_weighted_kde_evaluate(points: np.ndarray, samples: np.ndarray, 
                               weights: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Vectorized KDE evaluation for multiple points.
    
    Args:
        points: Array of points at which to evaluate the KDE
        samples: Array of sample points
        weights: Array of weights for each sample
        bandwidth: Kernel bandwidth parameter
        
    Returns:
        Array of KDE values at each point
    """
    results = np.zeros(len(points), dtype=np.float64)
    for i in prange(len(points)):
        results[i] = weighted_kde_evaluate(points[i], samples, weights, bandwidth)
    return results

def _round_for_cache(x: float) -> float:
    """Round a float value to the specified precision for cache keys."""
    # This approach prevents floating point precision issues
    return float(int(x * (10**CACHE_PRECISION)) / (10**CACHE_PRECISION))

class FastKDE:
    """
    Fast Kernel Density Estimation implementation with caching and optimizations.
    
    Attributes:
        data: Sample points (1D array)
        weights: Weight for each sample point
        bandwidth: Kernel bandwidth parameter
        tree: KD-tree for fast neighbor lookups
    """
    
    def __init__(self, data: np.ndarray, weights: Optional[np.ndarray] = None, 
                 bandwidth: Union[str, float] = 'silverman'):
        """
        Initialize the FastKDE estimator.
        
        Args:
            data: Sample points (will be flattened to 1D)
            weights: Weight for each sample point (defaults to uniform)
            bandwidth: Either 'silverman' for automatic bandwidth selection,
                      or a float specifying the bandwidth directly
        """
        # Ensure data is a 1D array
        self.data = np.asarray(data, dtype=np.float64).flatten()
        
        # Handle weights
        if weights is None:
            self.weights = np.ones_like(self.data, dtype=np.float64)
        else:
            # Ensure weights is a 1D array matching data length
            weights_array = np.asarray(weights, dtype=np.float64)
            if weights_array.size != self.data.size:
                raise ValueError(f"Weights size ({weights_array.size}) must match data size ({self.data.size})")
            self.weights = weights_array.flatten()
            
        # Normalize weights to sum to 1
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # If all weights are zero, use uniform weights
            self.weights = np.ones_like(self.data, dtype=np.float64) / len(self.data)
        
        # Set bandwidth
        if bandwidth == 'silverman':
            # Silverman's rule of thumb for bandwidth selection
            n = len(self.data)
            if n <= 1:
                self.bandwidth = 1.0  # Default for single point
            else:
                if weights is None:
                    # Standard deviation with small epsilon to prevent zero bandwidth
                    sigma = max(np.std(self.data), 1e-6)  
                else:
                    # Weighted mean and variance
                    weighted_mean = np.sum(self.weights * self.data)
                    weighted_var = np.sum(self.weights * (self.data - weighted_mean)**2)
                    sigma = max(np.sqrt(weighted_var), 1e-6)
                    # Effective sample size for weighted data
                    n_eff = 1 / np.sum(self.weights ** 2)
                    n = n_eff
                # Silverman's rule of thumb
                self.bandwidth = sigma * (4 / (3 * n)) ** 0.2
        else:
            # Use user-specified bandwidth with minimum threshold
            self.bandwidth = max(float(bandwidth), 1e-6)
            
        # Create KD-tree for nearest neighbor lookup (needs 2D array)
        self.tree = cKDTree(self.data[:, np.newaxis])
        
        # Cache for repeated evaluations (limited size)
        self._cache = {}
        self._max_cache_size = 1000
    
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the KDE at the specified points.
        
        Args:
            points: Points at which to evaluate the KDE
            
        Returns:
            KDE values at each point
        """
        # Ensure points is a 1D array
        points = np.asarray(points, dtype=np.float64).flatten()
        results = np.zeros(len(points), dtype=np.float64)
        
        # For small number of points, process individually with caching
        if len(points) <= 10:
            for i, x in enumerate(points):
                # Create cache key
                cache_key = _round_for_cache(x)
                
                # Check cache first
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                    continue
                    
                # Find points within 3*bandwidth for faster computation
                indices = self.tree.query_ball_point([x], 3 * self.bandwidth)[0]
                
                if indices:
                    # Only compute with relevant neighbors
                    relevant_data = self.data[indices]
                    relevant_weights = self.weights[indices]
                    result = weighted_kde_evaluate(
                        x, relevant_data, relevant_weights, self.bandwidth
                    )
                    results[i] = result
                    
                    # Cache result with rounded key
                    if len(self._cache) < self._max_cache_size:
                        self._cache[cache_key] = result
        else:
            # For larger point sets, process in chunks
            chunk_size = min(1000, len(points))
            for i in range(0, len(points), chunk_size):
                chunk = points[i:i+chunk_size]
                for j, x in enumerate(chunk):
                    cache_key = _round_for_cache(x)
                    if cache_key in self._cache:
                        results[i+j] = self._cache[cache_key]
                        continue
                        
                    indices = self.tree.query_ball_point([x], 3 * self.bandwidth)[0]
                    if indices:
                        relevant_data = self.data[indices]
                        relevant_weights = self.weights[indices]
                        result = weighted_kde_evaluate(
                            x, relevant_data, relevant_weights, self.bandwidth
                        )
                        results[i+j] = result
                        
                        if len(self._cache) < self._max_cache_size:
                            self._cache[cache_key] = result
        
        return results
    
    def logpdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute log probability density function at the specified points.
        
        Args:
            points: Points at which to evaluate the log PDF
            
        Returns:
            Log PDF values at each point
        """
        return np.log(np.maximum(self.evaluate(points), EPSILON))

    def __eq__(self, other: Any) -> bool:
        """Check if two KDEs are effectively equivalent"""
        if not isinstance(other, FastKDE):
            return False
        
        return (np.array_equal(self.data, other.data) and 
                np.array_equal(self.weights, other.weights) and
                self.bandwidth == other.bandwidth)


@njit(cache=True)
def _compute_ln_likelihood_kde_core(
    measurements: np.ndarray, 
    affected_flags: np.ndarray,  # 1 for affected (theta), 0 for non-affected (phi)
    bio_indices: np.ndarray,
    kde_data: np.ndarray,        # 2D array: [biomarker*2 + affected_flag, sample_idx]
    kde_weights: np.ndarray,     # 2D array: [biomarker*2 + affected_flag, sample_idx]
    bandwidths: np.ndarray       # 2D array: [biomarker, affected_flag]
) -> float:
    """
    Compute KDE log PDF efficiently using Numba.
    
    Args:
        measurements: Biomarker measurements
        affected_flags: 1 for affected (theta) state, 0 for non-affected (phi) state
        bio_indices: Biomarker indices
        kde_data: KDE sample points
        kde_weights: KDE weights
        bandwidths: Bandwidths for each biomarker and state
        
    Returns:
        Total log PDF value
    """
    total = 0.0
    
    for i in range(len(measurements)):
        idx = bio_indices[i]
        x = measurements[i]
        affected = affected_flags[i]
        
        # Determine KDE parameters based on affected state
        data_row = idx * 2 + affected    # Row index in kde_data/kde_weights
        bw = bandwidths[idx, affected]   # Get correct bandwidth
        
        data = kde_data[data_row]        # Sample points
        weights = kde_weights[data_row]  # Weights
        
        pdf = 0.0
        # Only process non-zero values (optimization)
        for j in range(len(data)):
            if j >= len(data) or data[j] == 0:  # End of data or padding
                break
            pdf += weights[j] * gaussian_kernel(x, data[j], bw)
        
        # Handle numerical stability
        if pdf < EPSILON:
            total += np.log(EPSILON)
        else:
            total += np.log(pdf)
    
    return total

def compute_ln_likelihood_kde_fast(
    measurements: np.ndarray, 
    S_n: np.ndarray, 
    biomarkers: np.ndarray, 
    k_j: int, 
    kde_dict: Dict[str, Dict[str, Union[FastKDE, np.ndarray]]]
) -> float:
    """
    Optimized KDE likelihood computation.
    
    Args:
        measurements: Biomarker measurements
        S_n: Stage thresholds
        biomarkers: Biomarker identifiers
        k_j: Stage value
        kde_dict: Dictionary of KDE objects for each biomarker
        
    Returns:
        Log likelihood value
    """
    # Convert to stage indicators (1 for affected, 0 for non-affected)
    affected_flags = (k_j >= S_n).astype(np.int32)
    
    # Map biomarkers to indices
    unique_bios = np.array(list(kde_dict.keys()))
    bio_to_idx = {b: i for i, b in enumerate(unique_bios)}
    bio_indices = np.array([bio_to_idx[b] for b in biomarkers], dtype=np.int32)
    
    # Calculate maximum data size needed - simplified since data is the same for theta and phi
    max_data_size = max(len(kde_dict[b]['theta_kde'].data) for b in unique_bios)
    
    # Pre-allocate arrays for Numba
    kde_data = np.zeros((len(unique_bios) * 2, max_data_size), dtype=np.float64)
    kde_weights = np.zeros((len(unique_bios) * 2, max_data_size), dtype=np.float64)
    bandwidths = np.zeros((len(unique_bios), 2), dtype=np.float64)
    
    # Fill arrays with data
    for i, b in enumerate(unique_bios):
        theta_kde = kde_dict[b]['theta_kde']
        phi_kde = kde_dict[b]['phi_kde']
        
        # Get data length (same for both theta and phi)
        data_len = len(theta_kde.data)
        
        # Phi data and weights (non-affected, index 0)
        kde_data[i*2, :data_len] = phi_kde.data
        kde_weights[i*2, :data_len] = phi_kde.weights
        
        # Theta data and weights (affected, index 1)
        kde_data[i*2+1, :data_len] = theta_kde.data
        kde_weights[i*2+1, :data_len] = theta_kde.weights
        
        # Store bandwidths
        bandwidths[i, 0] = phi_kde.bandwidth    # Non-affected (phi)
        bandwidths[i, 1] = theta_kde.bandwidth  # Affected (theta)
    
    # Compute log likelihood
    return _compute_ln_likelihood_kde_core(
        measurements.astype(np.float64),
        affected_flags,
        bio_indices,
        kde_data,
        kde_weights,
        bandwidths
    )

def get_initial_kde_estimates(
    data: pd.DataFrame
) -> Dict[str, Dict[str, FastKDE]]:
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
        
        # Skip biomarkers with too few measurements
        if len(biomarker_df) < 5:
            print(f"Warning: Skipping biomarker {biomarker} with only {len(biomarker_df)} measurements")
            continue
            
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
        
        # Create KDEs with the calculated weights - no need to store weights separately
        estimates[biomarker] = {
            'theta_kde': FastKDE(data=measurements, weights=theta_weights),
            'phi_kde': FastKDE(data=measurements, weights=phi_weights)
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
    theta_phi_current: Dict[str, Dict],
    disease_stages: np.ndarray,
    curr_order: int,
) -> Tuple[FastKDE, FastKDE]:
    """
    Update KDE estimates for a biomarker using EM with adaptive thresholds.
    
    Args:
        biomarker: Biomarker identifier
        participants: Participant IDs
        measurements: Measurements for this biomarker
        diseased: Disease status for participants
        stage_post: Stage posteriors from EM
        theta_phi_current: Current KDE estimates
        disease_stages: Disease stage values
        curr_order: Current biomarker order
        
    Returns:
        Updated theta_kde and phi_kde objects
    """
    # Ensure measurements is a 1D array
    measurements = np.asarray(measurements, dtype=np.float64).flatten()
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
    current_theta_kde = theta_phi_current[biomarker]['theta_kde']
    current_phi_kde = theta_phi_current[biomarker]['phi_kde']
    
    # Only update KDEs if weights changed significantly
    if np.mean(np.abs(theta_weights - current_theta_kde.weights)) < weight_change_threshold:
        theta_kde = current_theta_kde  # Reuse existing KDE
    else:
        theta_kde = FastKDE(data=measurements, weights=theta_weights)
    
    if np.mean(np.abs(phi_weights - current_phi_kde.weights)) < weight_change_threshold:
        phi_kde = current_phi_kde  # Reuse existing KDE
    else:
        phi_kde = FastKDE(data=measurements, weights=phi_weights)

    return theta_kde, phi_kde