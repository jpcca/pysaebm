import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
import logging
from collections import defaultdict 
from alabebm.utils.logging_utils import setup_logging 
import alabebm.utils.data_processing as data_utils 
import sys 

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

    # Compute means and standard deviations
    theta_mean = np.mean(affected_cluster) if affected_cluster else np.nan
    theta_std = np.std(affected_cluster) if affected_cluster else np.nan
    phi_mean = np.mean(
        non_affected_cluster) if non_affected_cluster else np.nan
    phi_std = np.std(non_affected_cluster) if non_affected_cluster else np.nan
    return theta_mean, theta_std, phi_mean, phi_std

def update_theta_phi_estimates(
    biomarker_data: Dict[str, Tuple[int, np.ndarray, np.ndarray, bool]],
    theta_phi_default: Dict[str, Dict[str, float]],
    participant_stage_posteriors: Dict[int, np.ndarray],
    diseased_stages:np.ndarray
    ) -> Dict[str, Dict[str, float]]:
    """Update theta and phi params using the soft K-means for all biomarkers."""
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
    diseased_stages: np.ndarray
    ) -> Tuple[float, Dict[int, np.ndarray]]:
    """Calculate the total log likelihood across all participants 
        and obtain participant_stage_posteriors
    """
    total_ln_likelihood = 0.0 
    # This is only for diseased participants
    participant_stage_posteriors = {}
    num_diseased_stages = len(diseased_stages)

    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        if participant in non_diseased_ids:
            # Non-diseased participant (fixed k=0)
            ln_likelihood = data_utils.compute_ln_likelihood(
                measurements, S_n, biomarkers, k_j = 0, theta_phi = theta_phi)
        else:
            # Diseased participant (sum over possible stages)
            ln_stage_likelihoods = np.array([
                data_utils.compute_ln_likelihood(
                    measurements, S_n, biomarkers, k_j = k_j, theta_phi=theta_phi
                ) for k_j in diseased_stages
            ])
            # Use log-sum-exp trick for numerical stability
            max_ln_likelihood = np.max(ln_stage_likelihoods)
            stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
            likelihood_sum = np.sum(stage_likelihoods)
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)

            # if likelihood_sum == 0:
            #     # Edge case: All stages have effectively zero likelihood
            #     normalized_probs = np.ones(num_diseased_stages) / num_diseased_stages
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
            participant_stage_posteriors[participant] = stage_likelihoods/likelihood_sum

        total_ln_likelihood += ln_likelihood
    return total_ln_likelihood, participant_stage_posteriors

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
    data_we_have['S_n'] = data_we_have['biomarker'].map(current_order_dict)

    participant_data = {}
    for participant, pdata in data_we_have.groupby('participant'):
        measurements = pdata['measurement'].values 
        S_n = pdata['S_n'].values 
        biomarkers = pdata['biomarker'].values  
        participant_data[participant] = (measurements, S_n, biomarkers)
    return participant_data

def metropolis_hastings_soft_kmeans(
    data_we_have: pd.DataFrame,
    iterations: int,
    n_shuffle: int,
) -> Tuple[List[Dict], List[float]]:
    """Metropolis-Hastings clustering algorithm."""
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

        # in each iteration, we have updated current_order_dict and theta_phi_estimates
        new_order = current_order.copy()
        data_utils.shuffle_order(new_order, n_shuffle)
        new_order_dict = dict(zip(biomarkers, new_order))

        # Update participant data with the new order dict
        participant_data = preprocess_participant_data(data_we_have, new_order_dict)

        # Obtain biomarker data
        biomarker_data = preprocess_biomarker_data(data_we_have, new_order_dict)

        ln_likelihood, participant_stage_posteriors = compute_total_ln_likelihood_and_stage_likelihoods(
                participant_data,
                non_diseased_ids,
                theta_phi_estimates,
                diseased_stages
        )

        # Now, update theta_phi_estimates using soft kmeans
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
        # it will definitly update at the first iteration
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
