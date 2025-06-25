from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from collections import defaultdict
from numba import njit
from pysaebm.utils.kmeans import get_two_clusters_with_kmeans
from pysaebm.utils.fast_kde import (
    get_initial_kde_estimates,
    compute_ln_likelihood_kde_fast,
    update_kde_for_biomarker_em
)

class PairwisePrefences:
    """
    Calculates pairwise preference weights from a list of partial orderings.
    
    This class is responsible for computing the weights 'w_ij' which represent the
    aggregated preference for item 'i' to appear before item 'j' across all
    provided partial orderings.
    """
    def __init__(self, ordering_array:List[List[str]]):
        """
        Initializes the class with the input orderings.

        Args:
            ordering_array: A list of lists, where each inner list is a
                            partial ordering of items (strings).
        """
        self.ordering_array: List[List[str]] = ordering_array
        self.unique_elements: Set[str] = set(
            [item for sublist in self.ordering_array for item in sublist])
        self.weights: Dict[Tuple[str, str], int] = defaultdict(int)

    def obtain_weights(self):
        """
        Computes the w_ij weights based on the input orderings.

        The weight w_ij is the sum of preferences across all orderings.
        For each input ordering, if 'i' comes before 'j', the score for (i, j)
        increases by 1, and the score for (j, i) decreases by 1.
        """
        weights = defaultdict(int)
        for ordering in self.ordering_array:
            for i in range(0, len(ordering) - 1):
                for j in range(i+1, len(ordering)):
                    # For a given pair (item1, item2) where item1 precedes item2
                    item1 = ordering[i]
                    item2 = ordering[j]
                    
                    # Increment weight for i preceding j
                    weights[(item1, item2)] += 1
                    # Decrement weight for j preceding i
                    weights[(item2, item1)] -= 1
        self.weights = weights

class MCMC:
    """
    Uses Metropolis-Hastings MCMC to sample a total ordering.

    This class implements the MCMC algorithm to draw a sample from the
    probability distribution P(σ) ∝ exp(-E(σ)), where the energy E(σ)
    is defined by the pairwise preference weights.
    """
    def __init__(
            self, ordering_array:List[List[str]], iterations:int=1000, seed:int=42, n_shuffle:int=2, rng: Optional[np.random.Generator] = None):
        """
        Initializes the MCMC sampler.

        Args:
            ordering_array: The list of partial orderings.
            iterations: The number of MCMC iterations to perform.
            seed: A random seed for reproducibility.
            n_shuffle: The number of items to swap in each proposal step.
                       The paper suggests a swap of 2 items.
        """
        # 1. Calculate the preference weights from the input data.
        pairwise_prefs = PairwisePrefences(ordering_array)
        pairwise_prefs.obtain_weights()
        self.weights = pairwise_prefs.weights
        
        self.unique_elements_list = list(pairwise_prefs.unique_elements)

        # 2. Set up the random number generator and create an initial random permutation.
        #    This corresponds to step 1 in Algorithm 1.
        if not rng:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng 
        # The result is a numpy array of strings
        self.initial_random_ordering = self.rng.permutation(self.unique_elements_list)
        
        self.iterations = iterations 
        self.n_shuffle = n_shuffle

    def obtain_energy(self, ordering:np.array) -> float:
        """
        Calculates the energy E(σ) of a given ordering.

        The energy is defined as E(σ) = -Σ w_ij * 1[i <_σ j], where 1[i <_σ j]
        is 1 if i precedes j in the ordering σ, and 0 otherwise.

        Args:
            ordering: A numpy array representing a total ordering σ.

        Returns:
            The calculated energy of the ordering.
        """
        total_sum = 0 
        for i in range(0, len(ordering) - 1):
            for j in range(i+1, len(ordering)):
                # Sum the weights for all pairs (i, j) where i precedes j
                total_sum += self.weights[(ordering[i], ordering[j])]
        return -total_sum 
    
    def shuffle_order(
            self, arr: np.ndarray, n_shuffle: int, rng: np.random.Generator) -> None:
        """
        Proposes a new ordering (σ') by shuffling elements in the current one.
        This is a proposal mechanism for the MCMC algorithm.
        """
        # (Your existing shuffle logic is valid and left unchanged)
        if n_shuffle <= 1:
            raise ValueError("n_shuffle must be >= 2 or =0")
        if n_shuffle > len(arr):
            raise ValueError("n_shuffle cannot exceed array length")
        if n_shuffle == 0:
            return 

        indices = rng.choice(len(arr), size=n_shuffle, replace=False)
        original_indices = indices.copy()
        
        while True:
            shuffled_indices = rng.permutation(original_indices)
            if not np.any(shuffled_indices == original_indices):
                break 
        arr[indices] = arr[shuffled_indices]

    def obtain_sample_ordering(self) -> np.ndarray:
        """
        Runs the Metropolis-Hastings MCMC sampler to get a final ordering.

        Returns:
            A numpy array representing the final sampled total ordering.
        """
        # Start with the initial random permutation 
        current_order = self.initial_random_ordering
        # Calculate its energy
        current_energy = self.obtain_energy(current_order) 

        # Loop for T iterations 
        for _ in range(self.iterations):
            # Propose a new ordering σ' by swapping elements 
            new_order = current_order.copy()
            self.shuffle_order(new_order, self.n_shuffle, self.rng)
            new_energy = self.obtain_energy(new_order)

            # Calculate the acceptance probability, α = min(1, P(σ')/P(σ)).
            # P(σ')/P(σ) = exp(-E(σ')) / exp(-E(σ)) = exp(E(σ) - E(σ'))
            prob_accept = min(1.0, np.exp(current_energy - new_energy))

            # Accept the new ordering with probability α 
            if self.rng.random() < prob_accept:
                current_order = new_order
                current_energy = new_energy
                
        self.final_ordering = current_order
        return self.final_ordering

def get_initial_theta_phi_estimates(
    data: pd.DataFrame,
    prior_n: float,
    prior_v: float,
) -> Dict[str, Dict[str, float]]:
    """
    Obtain initial theta and phi estimates (mean and standard deviation) for each biomarker.
    (get the clusters using seeded k-means (semi-supervised KMeans);
     estimate the parameters using conjugate priors
    )

    Args:
    data (pd.DataFrame): DataFrame containing participant data with columns 'participant', 
        'biomarker', 'measurement', and 'diseased'.
    prior_n (float):  Weak prior (not data-dependent)
    prior_v (float):  Weak prior (not data-dependent)

    Returns:
    Dict[str, Dict[str, float]]: A dictionary where each key is a biomarker name,
        and each value is another dictionary containing the means and standard deviations 
        for theta and phi of that biomarker, with keys 'theta_mean', 'theta_std', 'phi_mean', 
        and 'phi_std'.
    """
    # empty hashmap of dictionaries to store the estimates
    estimates = {}
    biomarkers = data.biomarker.unique()
    for biomarker in biomarkers:
        # Filter data for the current biomarker
        # reset_index is necessary here because we will use healthy_df.index later
        biomarker_df = data[data['biomarker']== biomarker].reset_index(drop=True)
        theta_measurements, phi_measurements, _ = get_two_clusters_with_kmeans(biomarker_df)
        # Use MLE to calculate the fallback (also to provide the m0 and s0_sq)
        fallback_params = {
            'theta_mean': np.mean(theta_measurements),
            'theta_std': np.std(theta_measurements, ddof=1),
            'phi_mean': np.mean(phi_measurements),
            'phi_std': np.std(phi_measurements, ddof=1),
        }
        theta_mean, theta_std, phi_mean, phi_std = compute_theta_phi_biomarker_conjugate_priors(
                theta_measurements, phi_measurements, fallback_params, prior_n, prior_v)
        estimates[biomarker] = {
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'phi_mean': phi_mean,
            'phi_std': phi_std
        }
    return estimates


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
    biomarker_data: Dict[str, Tuple[int, np.ndarray, np.ndarray, bool]],
    theta_phi_current: Dict[str, Dict[str, float]], # Current state’s θ/φ
    stage_likelihoods_posteriors: Dict[int, np.ndarray],
    disease_stages:np.ndarray,
    algorithm: str,
    prior_n: float,    # Weak prior (not data-dependent)
    prior_v: float,     # Weak prior (not data-dependent)
    weight_change_threshold: float,
    ) -> Dict[str, Dict[str, float]]:
    """Update theta and phi params for all biomarkers.

    Args:
        - algorithm (str): either 'conjugate_prior' or 'mle'
    """
    updated_params = defaultdict(dict)
    for biomarker, (
        curr_order, measurements, participants, diseased) in biomarker_data.items():
        dic = {'biomarker': biomarker}
        theta_phi_current_biomarker = theta_phi_current[biomarker]
        if algorithm not in ['conjugate_priors', "mle", 'em', 'kde']:
            raise ValueError('Algorithm should be chosen among conjugate_priors, em, and mle! Check your spelling!')
        if algorithm == 'em': 
            theta_mean, theta_std, phi_mean, phi_std = update_theta_phi_biomarker_em(
            participants,
            measurements,
            diseased,
            stage_likelihoods_posteriors,
            disease_stages,
            curr_order,
        )
        elif algorithm == 'kde':
            theta_weights, phi_weights = update_kde_for_biomarker_em(
                biomarker,
                participants,
                measurements,
                diseased,
                stage_likelihoods_posteriors,
                theta_phi_current,
                disease_stages,
                curr_order        
            )
        else:
            affected_cluster, non_affected_cluster = obtain_affected_and_non_clusters(
                participants,
                measurements,
                diseased,
                stage_likelihoods_posteriors,
                disease_stages,
                curr_order,
            )
            if algorithm == 'conjugate_priors':
                theta_mean, theta_std, phi_mean, phi_std = compute_theta_phi_biomarker_conjugate_priors(
                    affected_cluster, non_affected_cluster, theta_phi_current_biomarker, prior_n, prior_v)
            elif algorithm == 'mle':
                theta_mean, theta_std, phi_mean, phi_std = update_theta_phi_biomarker_mle(
                    affected_cluster, non_affected_cluster, theta_phi_current_biomarker)
        
        if algorithm == 'kde':
            updated_params[biomarker] = {
                'data': measurements,
                'theta_weights': theta_weights,
                'phi_weights': phi_weights,
            }
        else:
            updated_params[biomarker] = {
                'theta_mean': theta_mean,
                'theta_std': theta_std,
                'phi_mean': phi_mean,
                'phi_std': phi_std,
            }
    return updated_params

def update_theta_phi_biomarker_em(
    participants: np.ndarray,
    measurements: np.ndarray,
    diseased: np.ndarray,
    stage_likelihoods_posteriors: Dict[int, np.ndarray],
    disease_stages: np.ndarray,
    curr_order: int,
    ) -> Tuple[float, float, float, float]:
    """ Obtain biomarker's parameters using soft kmeans
    """
    # Obtain two responsibilites
    # Responsibilities of affected cluster
    # an array; each float means the prob of each measurement in affected cluster
    # Essentially, they are weights

    # Note that what we are doing here is different from GMM EM because we are not using 
    # p1 and p2 when obtaining responsibilities
    resp_affected = [
        sum(stage_likelihoods_posteriors[p][disease_stages >= curr_order]) if is_diseased else 0.0
        for p, is_diseased in zip(participants, diseased)
    ]

    resp_affected = np.array(resp_affected)
    resp_nonaffected = 1 - resp_affected

    sum_affected = max(np.sum(resp_affected), 1e-9)
    sum_nonaffected = max(np.sum(resp_nonaffected), 1e-9)

    # Weighted average
    theta_mean = np.sum(resp_affected*measurements)/sum_affected
    phi_mean = np.sum(resp_nonaffected*measurements)/sum_nonaffected

    # Weighted STD
    theta_std = np.sqrt(np.sum(resp_affected*(measurements - theta_mean)**2) / sum_affected)
    phi_std = np.sqrt(np.sum(resp_nonaffected*(measurements - phi_mean)**2) / sum_nonaffected)
    return theta_mean, theta_std, phi_mean, phi_std

def obtain_affected_and_non_clusters(
    participants: np.ndarray,
    measurements: np.ndarray,
    diseased: np.ndarray,
    stage_likelihoods_posteriors: Dict[int, np.ndarray],
    disease_stages: np.ndarray,
    curr_order: int,
    ) -> Tuple[List[float], List[float]]:

    """
    Obtain both the affected and non-affected clusters for a single biomarker.

    Args:
        participants (np.ndarray): Array of participant IDs.
        measurements (np.ndarray): Array of measurements for the biomarker.
        diseased (np.ndarray): Boolean array indicating whether each participant is diseased.
        stage_likelihoods_posteriors (Dict[int, np.ndarray]): Dictionary mapping participant IDs to their stage likelihoods.
        disease_stages (np.ndarray): Array of stages considered diseased.
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
                stage_likelihoods = stage_likelihoods_posteriors[p]
                affected_prob = np.sum(stage_likelihoods[disease_stages >= curr_order])
                non_affected_prob = np.sum(stage_likelihoods[disease_stages < curr_order])
                if affected_prob > non_affected_prob:
                    affected_cluster.append(m)
                elif affected_prob < non_affected_prob:
                    non_affected_cluster.append(m)
                else:
                    if np.random.random() > 0.5:
                        affected_cluster.append(m)
                    else:
                        non_affected_cluster.append(m)
    return affected_cluster, non_affected_cluster

def compute_theta_phi_biomarker_conjugate_priors(
    affected_cluster: List[float],
    non_affected_cluster: List[float],
    theta_phi_current_biomarker: Dict[str, float], # Current state’s θ/φ
    prior_n: float,
    prior_v: float
    ) -> Tuple[float, float, float, float]:
    """
    When data follows a normal distribution with unknown mean (μ) and unknown variance (σ²), 
    the normal-inverse gamma distribution serves as a conjugate prior for these parameters. 
    This means the posterior distribution will also be a normal-inverse gamma distribution after updating with observed data.

    Args:
        affected_cluster (List[float]): list of biomarker measurements
        non_affected_cluster (List[float]): list of biomarker measurements
        theta_phi_current_biomarker (Dict[str, float]): the current state's theta/phi for this biomarker
        prior_n (strength of belief in prior of mean), and prior_v (prior degree of freedom) are the weakly infomred priors. 

    Returns:
        Tuple[float, float, float, float]: Mean and standard deviation for affected (theta) and non-affected (phi) clusters.
    """
    # --- Affected Cluster (Theta) ---
    if len(affected_cluster) < 2:  # Fallback if cluster has 0 or 1 data points
        theta_mean = theta_phi_current_biomarker['theta_mean']
        theta_std = theta_phi_current_biomarker['theta_std']
    else:
        theta_mean, theta_std = estimate_params_exact(
            m0=theta_phi_current_biomarker['theta_mean'], 
            # m0=np.mean(affected_cluster),
            n0=prior_n, 
            # s0_sq = np.var(affected_cluster, ddof=1),
            s0_sq=theta_phi_current_biomarker['theta_std']**2, 
            v0=prior_v, 
            data=affected_cluster
        )
    
    # --- Non-Affected Cluster (Phi) ---
    if len(non_affected_cluster) < 2:  # Fallback if cluster has 0 or 1 data points
        phi_mean = theta_phi_current_biomarker['phi_mean']
        phi_std = theta_phi_current_biomarker['phi_std']
    else:
        phi_mean, phi_std = estimate_params_exact(
            m0=theta_phi_current_biomarker['phi_mean'], 
            # m0=np.mean(non_affected_cluster),
            n0=prior_n, 
            # s0_sq = np.var(non_affected_cluster, ddof=1),
            s0_sq=theta_phi_current_biomarker['phi_std']**2, 
            v0=prior_v, 
            data=non_affected_cluster
        )
    return theta_mean, theta_std, phi_mean, phi_std

def update_theta_phi_biomarker_mle(
    affected_cluster: List[float],
    non_affected_cluster: List[float],
    theta_phi_current_biomarker: Dict[str, float], # Current state’s θ/φ
    ) -> Tuple[float, float, float, float]:
    """
    maximum likelihood estimation (MLE)
    Treats parameters as fixed, unknown constants to be estimated.
    Relies only on observed data to compute estimates, ignoring prior information.

    Args:
        affected_cluster (List[float]): list of biomarker measurements
        non_affected_cluster (List[float]): list of biomarker measurements

    Returns:
        Tuple[float, float, float, float]: Mean and standard deviation for affected (theta) and non-affected (phi) clusters.
    """
    
    # Compute means and standard deviations
    theta_mean = np.mean(affected_cluster) if affected_cluster else theta_phi_current_biomarker['theta_mean']
    theta_std = np.std(affected_cluster, ddof=1) if len(affected_cluster) >= 2 else theta_phi_current_biomarker['theta_std']
    phi_mean = np.mean(
        non_affected_cluster) if non_affected_cluster else theta_phi_current_biomarker['phi_mean']
    phi_std = np.std(non_affected_cluster, ddof=1) if len(non_affected_cluster) >=2 else theta_phi_current_biomarker['phi_std']
    return theta_mean, theta_std, phi_mean, phi_std

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
    # Change the column of S_n inplace
    data_we_have = data_we_have.copy()
    data_we_have.loc[:, 'S_n'] = data_we_have['biomarker'].map(current_order_dict)

    participant_data = {}
    for participant, pdata in data_we_have.groupby('participant'):
        # Will be a numpy array
        measurements = pdata['measurement'].values 
        S_n = pdata['S_n'].values 
        biomarkers = pdata['biomarker'].values  
        participant_data[participant] = (measurements, S_n, biomarkers)
    return participant_data

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
    # Change the column of S_n inplace
    # Ensuring that we are explicitly modifying data_we_have and not an ambiguous copy.
    data_we_have = data_we_have.copy()
    data_we_have.loc[:, 'S_n'] = data_we_have['biomarker'].map(current_order_dict)

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
    algorithm:str,
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict],
    current_pi: np.ndarray,
    disease_stages: np.ndarray,
    bw_method: str
    ) -> Tuple[float, Dict[int, np.ndarray]]:
    """Calculate the total log likelihood across all participants 
        and obtain stage_likelihoods_posteriors
    """
    total_ln_likelihood = 0.0 
    # This is only for diseased participants
    stage_likelihoods_posteriors = {}
    # num_disease_stages = len(disease_stages)

    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        if participant in non_diseased_ids:
            # Non-diseased participant (fixed k=0)
            if algorithm == 'kde':
                ln_likelihood = compute_ln_likelihood_kde_fast(
                    measurements, S_n, biomarkers, k_j = 0, kde_dict = theta_phi, bw_method=bw_method
                )
            else:
                ln_likelihood = compute_ln_likelihood(
                    measurements, S_n, biomarkers, k_j = 0, theta_phi = theta_phi)
        else:
            # Diseased participant (sum over possible stages)
            if algorithm == 'kde':
                ln_stage_likelihoods = np.array([
                    compute_ln_likelihood_kde_fast(
                        measurements, S_n, biomarkers, k_j = k_j, kde_dict=theta_phi, bw_method=bw_method
                    ) + np.log(current_pi[k_j-1])
                    for k_j in disease_stages
                ])
            else:
                ln_stage_likelihoods = np.array([
                    compute_ln_likelihood(
                        measurements, S_n, biomarkers, k_j = k_j, theta_phi=theta_phi
                    ) + np.log(current_pi[k_j-1])
                    for k_j in disease_stages
                ])
            # Use log-sum-exp trick for numerical stability
            max_ln_likelihood = np.max(ln_stage_likelihoods)
            stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
            likelihood_sum = np.sum(stage_likelihoods)
            # Proof: https://hongtaoh.com/en/2024/12/14/log-sum-exp/
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)

            # if likelihood_sum == 0:
            #     # Edge case: All stages have effectively zero likelihood
            #     normalized_probs = np.ones(num_disease_stages) / num_disease_stages
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
            stage_likelihoods_posteriors[participant] = stage_likelihoods/likelihood_sum

        total_ln_likelihood += ln_likelihood
    return total_ln_likelihood, stage_likelihoods_posteriors

def obtain_unbaised_stage_likelihood_posteriors(
        algorithm:str,
        participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        theta_phi: Dict[str, Dict],
        current_pi: np.ndarray,
        bw_method: str
        ) -> Dict[int, np.ndarray]:
    """Obtain stage_likelihoods_posteriors while ignoring the diagnosis label or diseased or not. 
    """
    stage_likelihoods_posteriors = {}

    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        
        if algorithm == 'kde':
            ln_stage_likelihoods = np.array([
                compute_ln_likelihood_kde_fast(
                    measurements, S_n, biomarkers, k_j = k_j, kde_dict=theta_phi, bw_method=bw_method
                ) + np.log(current_pi[k_j-1])
                for k_j in range(0, len(theta_phi) + 1)
            ])
        else:
            ln_stage_likelihoods = np.array([
                compute_ln_likelihood(
                    measurements, S_n, biomarkers, k_j = k_j, theta_phi=theta_phi
                ) + np.log(current_pi[k_j-1])
                for k_j in range(0, len(theta_phi) + 1)
            ])
        # Use log-sum-exp trick for numerical stability
        max_ln_likelihood = np.max(ln_stage_likelihoods)
        stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
        likelihood_sum = np.sum(stage_likelihoods)

        stage_likelihoods_posteriors[participant] = stage_likelihoods/likelihood_sum

    return stage_likelihoods_posteriors

@njit
def _compute_ln_likelihood_core(measurements, mus, stds):
    """Core computation function optimized with Numba"""
    ln_likelihood = 0.0
    log_two_pi = np.log(2 * np.pi)
    two_times_pi = 2 * np.pi
    for i in range(len(measurements)):
        var = stds[i] ** 2
        diff = measurements[i] - mus[i]
        # likelihood *= np.exp(-diff**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
        # Log of normal PDF: ln(1/sqrt(2π*var) * exp(-diff²/2var))
        # = -ln(sqrt(2π*var)) - diff²/2var
        ln_likelihood += (-0.5 * (log_two_pi + np.log(var)) - diff**2 / (2 * var))
    return ln_likelihood

def compute_ln_likelihood(
    measurements: np.ndarray,
    S_n: np.ndarray,
    biomarkers: np.ndarray,
    k_j: int,
    theta_phi: Dict[str, Dict[str, float]]
) -> float:
    """
    Compute the log likelihood for given participant data.

    Args:
        measurements (np.ndarray): Array of measurement values.
        S_n (np.ndarray): Array of stage values (mapped from biomarkers).
        biomarkers (np.ndarray): Array of biomarker names.
        k_j (int): Current stage.
        theta_phi (Dict[str, Dict[str, float]]): Biomarker parameter dictionary.

    Returns:
        float: Log likelihood value.
    """
    mus = np.zeros(len(measurements))
    stds = np.zeros(len(measurements))
    affected = k_j >= S_n

    for i, (biomarker, is_affected) in enumerate(zip(biomarkers, affected)):
        params = theta_phi[biomarker]
        if is_affected:
            mus[i] = params['theta_mean']
            stds[i] = params['theta_std']
        else:
            mus[i] = params['phi_mean']
            stds[i] = params['phi_std']
    
    # Apply mask after mus and stds are computed
    valid_mask = (~np.isnan(measurements)) & (~np.isnan(mus)) & (stds > 0)
    measurements = measurements[valid_mask]
    mus = mus[valid_mask]
    stds = stds[valid_mask]

    return _compute_ln_likelihood_core(measurements, mus, stds)

def shuffle_order(arr: np.ndarray, n_shuffle: int, rng: np.random.Generator) -> None:

    """
    Randomly shuffle a specified number of elements in an array.

    Args:
    arr (np.ndarray): The array to shuffle elements in.
    n_shuffle (int): The number of elements to shuffle within the array.
    """
    # Validate input 
    if n_shuffle <= 1:
        raise ValueError("n_shuffle must be >= 2 or =0")
    if n_shuffle > len(arr):
        raise ValueError("n_shuffle cannot exceed array length")
    if n_shuffle == 0:
        return 

    # Select indices and extract elements
    indices = rng.choice(len(arr), size=n_shuffle, replace=False)
    original_indices = indices.copy()
    
    while True:
        shuffled_indices = rng.permutation(original_indices)
        # Full derangement: make sure no indice stays in its original place
        if not np.any(shuffled_indices == original_indices):
            break 
    arr[indices] = arr[shuffled_indices]

def obtain_most_likely_order_dic(all_current_accepted_order_dicts, burn_in, thining):
    """Obtain the most likely order based on all the accepted orders 
    Inputs:
        - all_current_accepted_order_dicts 
        - burn_in
        - thining
    Outputs:
        - a dictionary where key is biomarker and value is the most likely order for that biomarker
    """
    biomarker_stage_probability_df = get_biomarker_stage_probability(
        all_current_accepted_order_dicts, burn_in, thining)
    dic = {}
    assigned_stages = set()

    # Prioritize biomarkers with the highest maximum stage probability
    sorted_biomarkers = sorted(
        biomarker_stage_probability_df.index,
        key=lambda x: max(biomarker_stage_probability_df.loc[x]),
        reverse=True # Sort descending by highest probability
    )

    for biomarker in sorted_biomarkers:
        # Get probability array for this biomarker
        # The first number will be the prob of this biomarker in stage 1, etc. 
        prob_arr = np.array(biomarker_stage_probability_df.loc[biomarker])

        # Sort indices of probabilities in descending order
        sorted_indices = np.argsort(prob_arr)[::-1] + 1 # Stages are 1-based

        for stage in sorted_indices:
            if stage not in assigned_stages:
                dic[biomarker] = int(stage)
                assigned_stages.add(stage)
                break
        else:
            raise ValueError(
                f"Could not assign a unique stage for biomarker {biomarker}.")
    return dic

def get_biomarker_stage_probability(all_current_accepted_order_dicts, burn_in, thining):
    """filter through all_dicts using burn_in and thining 
    and for each biomarker, get probability of being in each possible stage

    Input:
        - all_current_accepted_order_dicts 
        - burn_in
        - thinning
    Output:
        - dff: a pandas dataframe where index is biomarker name, each col is each stage
        and each cell is the probability of that biomarker indicating that stage

        Note that in dff, its index follows the same order as data_we_have.biomarker.unique()

        Also note that it is guaranteed that the cols will be corresponding to state 1, 2, 3, ... in an asending order
    """
    df = pd.DataFrame(all_current_accepted_order_dicts)
    df = df[(df.index > burn_in) & (df.index % thining == 0)]
    # Create an empty list to hold dictionaries
    dict_list = []

    # biomarkers are in the same order as data_we_have.biomarker.unique()
    biomarkers = np.array(df.columns)

    # iterate through biomarkers
    for biomarker in biomarkers:
        dic = {"biomarker": biomarker}
        # get the frequency of biomarkers
        # value_counts will generate a Series where index is each cell's value
        # and the value is the frequency of that value
        stage_counts = df[biomarker].value_counts()
        # for each stage
        # note that df.shape[1] should be equal to num_biomarkers
        for i in range(1, df.shape[1] + 1):
            # get stage:prabability
            dic[i] = stage_counts.get(i, 0)/len(df)
        dict_list.append(dic)

    dff = pd.DataFrame(dict_list)
    dff.set_index(dff.columns[0], inplace=True)
    return dff