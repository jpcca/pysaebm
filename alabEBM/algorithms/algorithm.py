import numpy as np 
import pandas as pd 
import alabebm.utils.data_processing as data_utils 
from typing import List, Dict, Tuple
import logging 
from collections import defaultdict 

def metropolis_hastings(
    data_we_have: pd.DataFrame,
    iterations: int,
    n_shuffle: int,
    algorithm: str = 'conjugate_priors',
    prior_n: float = 1.0,    # Weak prior (not data-dependent)
    prior_v: float = 2.0     # Weak prior (not data-dependent)
) -> Tuple[List[Dict], List[float]]:
    """
    Perform Metropolis-Hastings sampling with conjugate priors to estimate biomarker orderings.

    Args:
        data_we_have (pd.DataFrame): Raw participant data.
        iterations (int): Number of iterations for the algorithm.
        n_shuffle (int): Number of swaps to perform when shuffling the order.
        algorithm (str): 'hard_kmeans', 'conjugate_priors', 'mle'
        prior_n (float):  Weak prior (not data-dependent)
        prior_v (float):  Weak prior (not data-dependent)

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
    current_theta_phi = theta_phi_default.copy()

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
        log_likelihoods.append(current_ln_likelihood)

        new_order = current_order.copy()
        data_utils.shuffle_order(new_order, n_shuffle)
        new_order_dict = dict(zip(biomarkers, new_order))

        """
        When we propose a new ordering, we want to calculate the total ln likelihood, which is 
        dependent on theta_phi_estimates, which are dependent on biomarker_data and stage_likelihoods_posterior,
        both of which are dependent on the ordering. 

        Therefore, we need to update participant_data, biomarker_data, stage_likelihoods_posterior
        and theta_phi_estimates before we can calculate the total ln likelihood associated with the new ordering
        """

        # Update participant data with the new order dict
        participant_data = data_utils.preprocess_participant_data(data_we_have, new_order_dict)

        """
        If conjugate priors or MLE, update theta_phi_estimates
        """
        if algorithm in ['conjugate_priors', 'mle']:

            biomarker_data = data_utils.preprocess_biomarker_data(data_we_have, new_order_dict)

            # Compute stage_likelihoods_posteriors using current theta_phi_estimates
            _, stage_likelihoods_posteriors = data_utils.compute_total_ln_likelihood_and_stage_likelihoods(
                participant_data,
                non_diseased_ids,
                current_theta_phi,
                diseased_stages
            )

            # Compute theta_phi_estimates based on new_order
            new_theta_phi = data_utils.update_theta_phi_estimates(
                biomarker_data,
                current_theta_phi, # Fallback uses current state’s θ/φ
                stage_likelihoods_posteriors,
                diseased_stages,
                algorithm = algorithm,
                prior_n = prior_n, 
                prior_v = prior_v,
            )

            # Recompute new_ln_likelihood using the new theta_phi_estimates
            new_ln_likelihood, _ = data_utils.compute_total_ln_likelihood_and_stage_likelihoods(
                participant_data,
                non_diseased_ids,
                new_theta_phi,
                diseased_stages
            )
        else:
            # If hard kmeans, it will use `theta_phi_estimates = theta_phi_default.copy()` defined above
            new_ln_likelihood, _ = data_utils.compute_total_ln_likelihood_and_stage_likelihoods(
                participant_data,
                non_diseased_ids,
                theta_phi_default,
                diseased_stages
            )

        # Compute acceptance probability
        delta = new_ln_likelihood - current_ln_likelihood
        prob_accept = 1.0 if delta > 0 else np.exp(delta)

        # Accept or reject the new state
        if np.random.rand() < prob_accept:
            current_order = new_order
            current_order_dict = new_order_dict
            current_ln_likelihood = new_ln_likelihood
            if algorithm != 'hard_kmeans':
                current_theta_phi = new_theta_phi
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
                # f"Current Theta and Phi Parameters: {theta_phi_estimates.items()} "
            )
    return all_accepted_orders, log_likelihoods