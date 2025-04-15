import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple, Dict, Union
from scipy.stats import mode

def get_two_clusters_with_kmeans(
    biomarker_df:pd.DataFrame, 
    max_attempt: int = 100, 
    seed = None) -> Tuple[np.ndarray, np.ndarray]:
    """get affected and nonaffected clusters for a biomarker using seeded k-means (semi-supervised KMeans)
    input: 
        - biomarker_df: a pd.dataframe of a specific biomarker
    output: 
        - A Tuple: two arrays containing the measurements of each cluster (affected, and nonaffected)
        # Note that it is guaranteed that both clusters have at least 2 elements; otherwise, the program will stop. 
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

    is_theta = (predictions == theta_cluster_idx).astype(int)

    assert len(is_theta) == len(measurements), "is_theta should have the same length as measurements"

    return theta_measurements, phi_measurements, is_theta