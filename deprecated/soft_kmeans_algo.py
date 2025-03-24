import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
import logging
from collections import defaultdict 
from alabebm.utils.logging_utils import setup_logging 
import alabebm.utils.data_processing as data_utils 
import sys 

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
    all_accepted_orders = []
    # This records all log likelihoods
    log_likelihoods = []

    for iteration in range(iterations):
        # floats are immutable, so no need to use .copy()
        log_likelihoods.append(current_ln_likelihood)

        # in each iteration, we have updated current_order_dict and theta_phi_estimates
        new_order = current_order.copy()
        data_utils.shuffle_order(new_order, n_shuffle)
        new_order_dict = dict(zip(biomarkers, new_order))

        # I am changing the col of S_n in both preprocess_participant_data
        # and preprocess_biomarker_data just to be safe

        # Update participant data with the new order dict
        participant_data = data_utils.preprocess_participant_data(data_we_have, new_order_dict)
        # Obtain biomarker data
        biomarker_data = data_utils.preprocess_biomarker_data(data_we_have, new_order_dict)

        theta_phi_estimates = theta_phi_default.copy()

        # Compute stage_likelihoods_posteriors using current theta_phi_estimates
        _, stage_likelihoods_posteriors = data_utils.compute_total_ln_likelihood_and_stage_likelihoods(
            participant_data,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages
        )

        # Compute new_theta_phi_estimates based on new_order
        new_theta_phi_estimates = data_utils.update_theta_phi_estimates(
            biomarker_data,
            theta_phi_estimates,
            stage_likelihoods_posteriors,
            diseased_stages
        )

        # Recompute new_ln_likelihood using new_theta_phi_estimates
        new_ln_likelihood_new_theta_phi, _ = data_utils.compute_total_ln_likelihood_and_stage_likelihoods(
            participant_data,
            non_diseased_ids,
            new_theta_phi_estimates,
            diseased_stages
        )

        delta = new_ln_likelihood_new_theta_phi - current_ln_likelihood
        prob_accept = 1.0 if delta > 0 else np.exp(delta)
        # Proof:
        # prob_accept = np.exp(ln_likelihood - current_ln_likelihood)
        # np.exp(a)/np.exp(b) = np.exp(a - b)
        # if a > b, then np.exp(a - b) > 1

        # Accept or reject the new state
        if np.random.rand() < prob_accept:
            current_order = new_order
            current_order_dict = new_order_dict
            current_ln_likelihood = new_ln_likelihood_new_theta_phi
            theta_phi_estimates = new_theta_phi_estimates
            acceptance_count += 1

        all_accepted_orders.append(current_order_dict.copy())

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
    return all_accepted_orders, log_likelihoods
