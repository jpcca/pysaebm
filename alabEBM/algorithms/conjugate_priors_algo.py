import numpy as np 
import pandas as pd 
import alabEBM.utils.data_processing as data_utils 
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

def update_theta_phi_estimates(
    biomarker_data: Dict[str, Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], 
    theta_phi_default: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Update theta (θ) and phi (φ) parameters for all biomarkers using conjugate priors.

    Args:
        biomarker_data (Dict): Dictionary containing biomarker data. Keys are biomarker names, and values
            are tuples of (curr_order, measurements, participants, diseased, affected).
        theta_phi_default (Dict): Default values for theta and phi parameters for each biomarker.

    Returns:
        Dict[str, Dict[str, float]]: Updated theta and phi parameters for each biomarker.

    Notes:
        - If there is only one observation or no observations at all, the function resorts to the default
          values provided in `theta_phi_default`.
        - This situation can occur if, for example, a biomarker indicates a stage of (num_biomarkers),
          but all participants' stages are smaller than that stage. In such cases, the biomarker is not
          affected for any participant, and default values are used.
    """
    updated_params = defaultdict(dict)
    for biomarker, (
        curr_order, measurements, participants, diseased, affected) in biomarker_data.items():
        theta_mean = theta_phi_default[biomarker]['theta_mean']
        theta_std = theta_phi_default[biomarker]['theta_std']
        phi_mean = theta_phi_default[biomarker]['phi_mean']
        phi_std = theta_phi_default[biomarker]['phi_std']

        for affected_bool in [True, False]:
            measurements_of_affected_bool = measurements[affected == affected_bool]
            if len(measurements_of_affected_bool) > 1:
                s0_sq = np.var(measurements_of_affected_bool, ddof=1)
                m0 = np.mean(measurements_of_affected_bool)
                mu_estimate, std_estimate = estimate_params_exact(
                    m0=m0, n0=1, s0_sq=s0_sq, v0=1, data=measurements_of_affected_bool)
                if affected_bool:
                    theta_mean = mu_estimate
                    theta_std = std_estimate
                else:
                    phi_mean = mu_estimate
                    phi_std = std_estimate
            
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
    participant_stages: np.ndarray
) -> Dict[str, Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Preprocess raw participant data into a structured format for biomarker analysis.

    Args:
        data_we_have (pd.DataFrame): Raw participant data.
        current_order_dict (Dict): Mapping of biomarkers to their current order (stages).
        participant_stages (np.ndarray): Array of disease stages for each participant.

    Returns:
        Dict[str, Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: Preprocessed biomarker data.
            Keys are biomarker names, and values are tuples of (curr_order, measurements, participants, diseased, affected).
    """
    # This modifies data source in-place
    data_we_have['S_n'] = data_we_have['biomarker'].map(current_order_dict)
    participant_stage_dic = dict(
        zip(np.arange(0, len(participant_stages)), participant_stages))
    data_we_have['k_j'] = data_we_have['participant'].map(participant_stage_dic)
    data_we_have['affected'] = data_we_have['k_j'] >= data_we_have['S_n']

    biomarker_data = {}
    for biomarker, bdata in data_we_have.groupby('biomarker'):
        curr_order = current_order_dict[biomarker]
        measurements = bdata['measurement'].values 
        participants = bdata['participant'].values  
        diseased = bdata['diseased'].values
        affected = bdata['affected'].values
        biomarker_data[biomarker] = (curr_order, measurements, participants, diseased, affected)
    return biomarker_data

def calculate_all_participant_ln_likelihood_and_update_participant_stages(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict[str, float]],
    diseased_stages: np.ndarray,
    participant_stages: np.ndarray
) -> float:
    """
    Calculate the total log likelihood across all participants and update their disease stages.

    Args:
        participant_data (Dict): Dictionary containing participant data. Keys are participant IDs, and values
            are tuples of (measurements, S_n, biomarkers).
        non_diseased_ids (np.ndarray): Array of participant IDs who are non-diseased.
        theta_phi (Dict): Theta and phi parameters for each biomarker.
        diseased_stages (np.ndarray): Array of possible disease stages.
        participant_stages (np.ndarray): Array of current disease stages for each participant.

    Returns:
        float: Total log likelihood across all participants.
    """
    total_ln_likelihood = 0.0 
    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        if participant in non_diseased_ids:
            likelihood = data_utils.compute_likelihood(
                measurements, S_n, biomarkers, k_j = 0, theta_phi = theta_phi)
        else:
            stage_likelihoods = np.array([
                data_utils.compute_likelihood(
                    measurements, S_n, biomarkers, k_j = k_j, theta_phi=theta_phi
                ) for k_j in diseased_stages
            ])
            epsilon = 1e-10
            likelihood_sum = np.sum(stage_likelihoods)
            if likelihood_sum == 0:
                participant_stages[participant] = np.random.choice(diseased_stages)
                likelihood_sum = epsilon
            else:
                normalized_probs = stage_likelihoods/likelihood_sum
                participant_stages[participant] = np.random.choice(diseased_stages, p=normalized_probs)
            likelihood = likelihood_sum / len(diseased_stages)
        total_ln_likelihood += np.log(likelihood)
    return total_ln_likelihood

def preprocess_participant_data(
    data_we_have: pd.DataFrame, 
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Preprocess participant data into NumPy arrays for efficient computation.

    Args:
        data_we_have (pd.DataFrame): Raw participant data.

    Returns:
        Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]: A dictionary where keys are participant IDs,
            and values are tuples of (measurements, S_n, biomarkers).
    """

    participant_data = {}
    for participant, pdata in data_we_have.groupby('participant'):
        measurements = pdata['measurement'].values 
        S_n = pdata['S_n'].values 
        biomarkers = pdata['biomarker'].values  
        participant_data[participant] = (measurements, S_n, biomarkers)
    return participant_data

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

    # Initiate random participant stages
    participant_stages = np.zeros(n_participants)
    for idx in range(n_participants):
        if idx not in non_diseased_ids:
            participant_stages[idx] = np.random.randint(1, len(diseased_stages) + 1)

    for iteration in range(iterations):
        log_likelihoods.append(current_ln_likelihood)

        new_order = current_order.copy()
        data_utils.shuffle_order(new_order, n_shuffle)
        new_order_dict = dict(zip(biomarkers, new_order))

        biomarker_data = preprocess_biomarker_data(
            data_we_have, new_order_dict, participant_stages)

        # Update participant data based on the new order
        # Update data_we_have based on the new order and the updated participant_stages
        participant_data = preprocess_participant_data(data_we_have)
        
        ln_likelihood = calculate_all_participant_ln_likelihood_and_update_participant_stages(
            participant_data,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages,
            participant_stages
        )

        max_likelihood = max(ln_likelihood, current_ln_likelihood)
        prob_accept = np.exp(
            (ln_likelihood - max_likelihood) -
            (current_ln_likelihood - max_likelihood)
        )

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
            # Update theta and phi parameters for all biomarkers
            # We basically need the original raw data and the updated affected col 
            theta_phi_estimates = update_theta_phi_estimates(
                biomarker_data, 
                theta_phi_default
            ) 
        
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