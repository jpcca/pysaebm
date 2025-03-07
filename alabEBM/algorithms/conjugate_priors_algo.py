import numpy as np 
import pandas as pd 
import alabebm.utils.data_processing as data_utils 
from . import soft_kmeans_algo as sk 
from typing import List, Dict, Tuple
import logging 
from collections import defaultdict 

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

def compute_theta_phi_biomarker(
    participants: np.ndarray,
    measurements: np.ndarray,
    diseased: np.ndarray,
    participant_stage_posteriors: Dict[int, np.ndarray],
    diseased_stages: np.ndarray,
    curr_order: int,
    ) -> Tuple[float, float, float, float]:
    """
    Compute mean and std for both the affected and non-affected clusters for a single biomarker.

    Args:
        participants (np.ndarray): Array of participant IDs.
        measurements (np.ndarray): Array of measurements for the biomarker.
        diseased (np.ndarray): Boolean array indicating whether each participant is diseased.
        participant_stage_posteriors (Dict[int, np.ndarray]): Dictionary mapping participant IDs to their stage likelihoods.
        diseased_stages (np.ndarray): Array of stages considered diseased.
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
                stage_likelihoods = participant_stage_posteriors[p]
                affected_prob = np.sum(stage_likelihoods[diseased_stages >= curr_order])
                non_affected_prob = np.sum(stage_likelihoods[diseased_stages < curr_order])
                if affected_prob > non_affected_prob:
                    affected_cluster.append(m)
                elif affected_prob < non_affected_prob:
                    non_affected_cluster.append(m)
                else:
                    if np.random.random() > 0.5:
                        affected_cluster.append(m)
                    else:
                        non_affected_cluster.append(m)
                        
    # np.var won't make sense if there is only one participant
    if len(affected_cluster) <= 1:
        theta_mean, theta_std = np.nan, np.nan 
    else:
        s0_sq = np.var(affected_cluster, ddof=1)
        m0 = np.mean(affected_cluster)
        theta_mean, theta_std = estimate_params_exact(
            m0=m0, n0=1, s0_sq=s0_sq, v0=1, data=affected_cluster)
    if len(non_affected_cluster) <= 1:
        phi_mean, phi_std = np.nan, np.nan 
    else:
        s0_sq = np.var(non_affected_cluster, ddof=1)
        m0 = np.mean(non_affected_cluster)
        phi_mean, phi_std = estimate_params_exact(
            m0=m0, n0=1, s0_sq=s0_sq, v0=1, data=non_affected_cluster)
    return theta_mean, theta_std, phi_mean, phi_std
        
def update_theta_phi_estimates(
    biomarker_data: Dict[str, Tuple[int, np.ndarray, np.ndarray, bool]],
    theta_phi_default: Dict[str, Dict[str, float]],
    participant_stage_posteriors: Dict[int, np.ndarray],
    diseased_stages:np.ndarray
    ) -> Dict[str, Dict[str, float]]:
    """Update theta and phi params using the conjugate priors for all biomarkers."""
    updated_params = defaultdict(dict)
    for biomarker, (
        curr_order, measurements, participants, diseased) in biomarker_data.items():
        dic = {'biomarker': biomarker}
        theta_phi_default_biomarker = theta_phi_default[biomarker]
        theta_mean, theta_std, phi_mean, phi_std = compute_theta_phi_biomarker(
            participants,
            measurements,
            diseased,
            participant_stage_posteriors,
            diseased_stages,
            curr_order,
        ) 
        if theta_std == 0 or np.isnan(theta_std):
            theta_mean = theta_phi_default_biomarker['theta_mean']
            theta_std = theta_phi_default_biomarker['theta_std']
        if phi_std == 0 or np.isnan(phi_std):
            phi_mean = theta_phi_default_biomarker['phi_mean']
            phi_std = theta_phi_default_biomarker['phi_std']
        updated_params[biomarker] = {
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'phi_mean': phi_mean,
            'phi_std': phi_std,
        }
    return updated_params

def metropolis_hastings_conjugate_priors(
    data_we_have: pd.DataFrame,
    iterations: int,
    n_shuffle: int
) -> Tuple[List[Dict], List[float]]:
    """
    Perform Metropolis-Hastings sampling with conjugate priors to estimate biomarker orderings.

    Args:
        data_we_have (pd.DataFrame): Raw participant data.
        iterations (int): Number of iterations for the algorithm.
        n_shuffle (int): Number of swaps to perform when shuffling the order.

    Returns:
        Tuple[List[Dict], List[float]]: 
            - List of accepted biomarker orderings at each iteration.
            - List of log likelihoods at each iteration.
    """
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_stages = len(biomarkers) + 1
    diseased_stages = np.arange(start=1, stop=n_stages, step=1)
    non_diseased_ids = data_we_have.loc[data_we_have.diseased == False].participant.unique()

    theta_phi_default = data_utils.get_theta_phi_estimates(data_we_have)
    theta_phi_estimates = theta_phi_default.copy()

    # initialize an ordering and likelihood
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

        new_order = current_order.copy()
        data_utils.shuffle_order(new_order, n_shuffle)
        new_order_dict = dict(zip(biomarkers, new_order))

        # Update participant data with the new order dict
        participant_data = sk.preprocess_participant_data(data_we_have, new_order_dict)

        # Obtain biomarker data
        biomarker_data = sk.preprocess_biomarker_data(data_we_have, new_order_dict)

        ln_likelihood, participant_stage_posteriors = sk.compute_total_ln_likelihood_and_stage_likelihoods(
                participant_data,
                non_diseased_ids,
                theta_phi_estimates,
                diseased_stages
        )

        # Now, update theta_phi_estimates using conjugate priors
        # based on the updated participant_stage_posteriors
        theta_phi_estimates = update_theta_phi_estimates(
            biomarker_data,
            theta_phi_default,
            participant_stage_posteriors,
            diseased_stages
        )

        delta = ln_likelihood - current_ln_likelihood
        # Compute acceptance probability safely
        if delta > 0:
            prob_accept = 1.0  # Always accept improvements
        else:
            prob_accept = np.exp(delta)  # Only exponentiate negative deltas

        # prob_accept = np.exp(ln_likelihood - current_ln_likelihood)
        # np.exp(a)/np.exp(b) = np.exp(a - b)
        # if a > b, then np.exp(a - b) > 1

        # Accept or reject 
        if np.random.rand() < prob_accept:
            current_order = new_order 
            current_ln_likelihood = ln_likelihood
            current_order_dict = new_order_dict 
            acceptance_count += 1
        
        all_orders.append(current_order_dict)

        # Log progress
        if (iteration + 1) % max(10, iterations // 10) == 0:
            acceptance_ratio = 100 * acceptance_count / (iteration + 1)
            logging.info(
                f"Iteration {iteration + 1}/{iterations}, "
                f"Acceptance Ratio: {acceptance_ratio:.2f}%, "
                f"Log Likelihood: {current_ln_likelihood:.4f}, "
                f"Current Accepted Order: {current_order_dict.values()}, "
                f"Current Theta and Phi Parameters: {theta_phi_estimates.items()} "
            )
    return all_orders, log_likelihoods