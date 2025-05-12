import numpy as np 
import pandas as pd 
import pysaebm.utils.data_processing as data_utils 
import pysaebm.utils.ucl_gmm as gmm
from typing import List, Dict, Tuple
import logging 

def metropolis_hastings(
    data_we_have: pd.DataFrame,
    iterations: int,
    n_shuffle: int,
    algorithm: str,
    prior_n: float,
    prior_v: float,
    weight_change_threshold: float = 0.01,
    bw_method: str = 'scott'
) -> Tuple[List[Dict], List[float], Dict[str, Dict], Dict[int, np.ndarray]]:
    """
    Perform Metropolis-Hastings sampling with conjugate priors to estimate biomarker orderings.

    Args:
        data_we_have (pd.DataFrame): Raw participant data.
        iterations (int): Number of iterations for the algorithm.
        n_shuffle (int): Number of swaps to perform when shuffling the order.
        algorithm (str): 'hard_kmeans', 'conjugate_priors', 'mle', 'em'
        prior_n (strength of belief in prior of mean)
        prior_v (prior degree of freedom) are the weakly infomred priors. 
        weight_change_threshold (float): Threshold for kde weights (if np.mean(new_weights - old_weights)) > threshold, then recalculate
            otherwise use the new kde and weights
        bw_method (str): bandwidth selection method in kde

    Returns:
        Tuple[List[Dict], List[float], Dict[str, Dict], Dict[int, np.ndarray]]: 
            - List of accepted biomarker orderings at each iteration.
            - List of log likelihoods at each iteration.
            - Final theta phi estimates
            - Stage likelihood posterior 
    """
    biomarkers = data_we_have.biomarker.unique()
    n_stages = len(biomarkers) + 1
    disease_stages = np.arange(start=1, stop=n_stages, step=1)
    n_disease_stages = n_stages - 1
    non_diseased_ids = data_we_have.loc[data_we_have.diseased == False].participant.unique()

    if algorithm == 'kde':
        theta_phi_default = data_utils.get_initial_kde_estimates(data_we_have)
    elif algorithm in ['gmm', 'gmm_dm_prior']:
        theta_phi_default = gmm.get_gmm_theta_phi_estimates(data_we_have)
    else:
        theta_phi_default = data_utils.get_initial_theta_phi_estimates(data_we_have, prior_n, prior_v)
    
    current_theta_phi = theta_phi_default.copy()

    # initialize an ordering and likelihood
    current_order = np.random.permutation(np.arange(1, n_stages))
    current_order_dict = dict(zip(biomarkers, current_order))
    current_ln_likelihood = -np.inf
    alpha_prior = [1.0]* (n_disease_stages)
    if algorithm in ['gmm']:
        current_pi = np.array([1/n_disease_stages] * n_disease_stages)
    else:
        current_pi = np.random.dirichlet(alpha_prior) # Sample from uniform dirichlet dist.
    current_stage_post = {}
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
        if algorithm not in ['hard_kmeans', 'gmm', 'gmm_dm_prior']:

            biomarker_data = data_utils.preprocess_biomarker_data(data_we_have, new_order_dict)

            # --- Compute stage posteriors with OLD θ/φ ---
            # Only diseased participants have stage likelihoods 
            _, stage_post_old = data_utils.compute_total_ln_likelihood_and_stage_likelihoods(
                algorithm,
                participant_data,
                non_diseased_ids,
                current_theta_phi,
                current_pi,
                disease_stages,
                bw_method
            )

            # Compute the new theta_phi_estimates based on new_order
            new_theta_phi = data_utils.update_theta_phi_estimates(
                biomarker_data,
                current_theta_phi, # Fallback uses current state’s θ/φ
                stage_post_old,
                disease_stages,
                algorithm = algorithm,
                prior_n = prior_n, 
                prior_v = prior_v,
                weight_change_threshold = weight_change_threshold
            )

            # Recompute new_ln_likelihood using the new theta_phi_estimates
            new_ln_likelihood, stage_post_new = data_utils.compute_total_ln_likelihood_and_stage_likelihoods(
                algorithm,
                participant_data,
                non_diseased_ids,
                new_theta_phi,
                current_pi,
                disease_stages,
                bw_method
            )

        else:
            # If hard kmeans or gmm, it will use `current_theta_phi = theta_phi_default.copy()` defined above
            new_ln_likelihood, stage_post_new = data_utils.compute_total_ln_likelihood_and_stage_likelihoods(
                algorithm,
                participant_data,
                non_diseased_ids,
                current_theta_phi,
                current_pi,
                disease_stages,
                bw_method
            )

        # Compute acceptance probability
        delta = new_ln_likelihood - current_ln_likelihood
        prob_accept = 1.0 if delta > 0 else np.exp(delta)

        # Accept or reject the new state
        if np.random.rand() < prob_accept:
            current_order = new_order
            current_order_dict = new_order_dict
            current_ln_likelihood = new_ln_likelihood
            current_stage_post = stage_post_new
            if algorithm not in ['hard_kmeans', 'gmm', 'gmm_dm_prior']:
                current_theta_phi = new_theta_phi
            acceptance_count += 1

            if algorithm not in ['gmm', 'conjugate_priors']:
                stage_counts = np.zeros(n_disease_stages)
                # participant, array of stage likelihoods
                for p, stage_probs in stage_post_new.items():
                    stage_counts += stage_probs # Soft counts 
                current_pi = np.random.dirichlet(alpha_prior + stage_counts)

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
        
    return all_accepted_orders, log_likelihoods, current_theta_phi, current_stage_post, current_pi