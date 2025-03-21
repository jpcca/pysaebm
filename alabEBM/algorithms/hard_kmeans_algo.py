import numpy as np
import numba 
import pandas as pd 
from typing import List, Dict, Tuple
import alabebm.utils.data_processing as data_utils 
from . import soft_kmeans_algo as sk 
import logging

def calculate_all_participant_ln_likelihood(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict[str, float]],
    diseased_stages: np.ndarray
    ) -> float:
    """Calculate the total log likelihood across all participants."""
    total_ln_likelihood = 0.0 
    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        if participant in non_diseased_ids:
            ln_likelihood = data_utils.compute_ln_likelihood(
                measurements, S_n, biomarkers, k_j = 0, theta_phi = theta_phi
            )
        else:
            ln_stage_likelihoods = [
                data_utils.compute_ln_likelihood(
                    measurements, S_n, biomarkers, k_j = k_j, theta_phi=theta_phi
                ) for k_j in diseased_stages
            ]
            # Use log-sum-exp trick for numerical stability
            max_ln_likelihood = np.max(ln_stage_likelihoods)
            stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
            likelihood_sum = np.sum(stage_likelihoods)
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)
            
        total_ln_likelihood += ln_likelihood

    return total_ln_likelihood

def metropolis_hastings_hard_kmeans(
    data_we_have: pd.DataFrame,
    iterations: int, 
    n_shuffle: int
    ) -> List[Dict]:
    """Metropolis-Hastings clustering algorithm."""

    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_stages = len(biomarkers) + 1

    diseased_stages = np.arange(1, n_stages)
    non_diseased_ids = data_we_have.loc[data_we_have.diseased == False].participant.unique()

    theta_phi_default = data_utils.get_theta_phi_estimates(data_we_have)

    logging.info(f"Default Theta and Phi Parameters: {theta_phi_default.items()} ")

    current_order = np.random.permutation(np.arange(1, n_stages))
    current_order_dict = dict(zip(biomarkers, current_order))
    current_ln_likelihood = -np.inf
    acceptance_count = 0
    # Note that this records only the current accepted orders in each iteration
    all_orders = []
    # This records all log likelihoods
    log_likelihoods = []

    for iteration in range(iterations):
        log_likelihoods.append(current_ln_likelihood)
        # Suffle the order 
        # Note that copy here is necessary because without it, each iteration is 
        # shuffling the order in the last iteration. 
        # With copy, we can ensure that the current state remains unchanged until
        # the proposed state is accepted.  

        new_order = current_order.copy()
        data_utils.shuffle_order(new_order, n_shuffle)
        new_order_dict = dict(zip(biomarkers, new_order))

        # Update participant data with the new order dict
        participant_data = sk.preprocess_participant_data(data_we_have, new_order_dict)

        # Calculate likelihoods
        ln_likelihood = calculate_all_participant_ln_likelihood(
            participant_data, non_diseased_ids, theta_phi_default, diseased_stages
        )
        
        delta = ln_likelihood - current_ln_likelihood
        prob_accept = 1.0 if delta > 0 else np.exp(delta)
        
        # it will definitly update at the first iteration
        if np.random.rand() < prob_accept:
            current_order = new_order 
            current_ln_likelihood = ln_likelihood
            current_order_dict = new_order_dict 
            acceptance_count += 1
        
        all_orders.append(current_order_dict.copy())

        # Log progress
        if (iteration + 1) % max(10, iterations // 10) == 0:
            acceptance_ratio = 100 * acceptance_count / (iteration + 1)
            logging.info(
                f"Iteration {iteration + 1}/{iterations}, "
                f"Acceptance Ratio: {acceptance_ratio:.2f}%, "
                f"Log Likelihood: {current_ln_likelihood:.4f}, "
                f"Current Accepted Order: {current_order_dict.values()}, "
            )
    return all_orders, log_likelihoods