import numpy as np
from . import utils

def calculate_soft_kmeans_for_biomarker(
        data,
        biomarker,
        order_dict,
        n_participants,
        non_diseased_participants,
        hashmap_of_normalized_stage_likelihood_dicts,
        diseased_stages,
        seed=None
):
    """
    Calculate mean and std for both the affected and non-affected clusters for a single biomarker.

    Parameters:
        data (pd.DataFrame): The data containing measurements.
        biomarker (str): The biomarker to process.
        order_dict (dict): Dictionary mapping biomarkers to their order.
        n_participants (int): Number of participants in the study.
        non_diseased_participants (list): List of non-diseased participants.
        hashmap_of_normalized_stage_likelihood_dicts (dict): Hash map of 
            dictionaries containing stage likelihoods for each participant.
        diseased_stages (list): List of diseased stages.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Means and standard deviations for affected and non-affected clusters.
    """
    if seed is not None:
        # Set the seed for numpy's random number generator
        rng = np.random.default_rng(seed)
    else:
        rng = np.random 

    # DataFrame for this biomarker
    biomarker_df = data[
        data['biomarker'] == biomarker].reset_index(
            drop=True).sort_values(
                by = 'participant', ascending = True)
    # Extract measurements
    measurements = np.array(biomarker_df['measurement'])

    this_biomarker_order = order_dict[biomarker]

    affected_cluster = []
    non_affected_cluster = []

    for p in range(n_participants):
        if p in non_diseased_participants:
            non_affected_cluster.append(measurements[p])
        else:
            if this_biomarker_order == 1:
                affected_cluster.append(measurements[p])
            else:
                normalized_stage_likelihood_dict = hashmap_of_normalized_stage_likelihood_dicts[
                    p]
                # Calculate probabilities for affected and non-affected states
                affected_prob = sum(
                    normalized_stage_likelihood_dict[s] for s in diseased_stages if s >= this_biomarker_order
                )
                non_affected_prob = sum(
                    normalized_stage_likelihood_dict[s] for s in diseased_stages if s < this_biomarker_order
                )
                if affected_prob > non_affected_prob:
                    affected_cluster.append(measurements[p])
                elif affected_prob < non_affected_prob:
                    non_affected_cluster.append(measurements[p])
                else:
                    # Assign to either cluster randomly if probabilities are equal
                    if rng.random() > 0.5:
                        affected_cluster.append(measurements[p])
                    else:
                        non_affected_cluster.append(measurements[p])

    # Compute means and standard deviations
    theta_mean = np.mean(affected_cluster) if affected_cluster else np.nan
    theta_std = np.std(affected_cluster) if affected_cluster else np.nan
    phi_mean = np.mean(
        non_affected_cluster) if non_affected_cluster else np.nan
    phi_std = np.std(non_affected_cluster) if non_affected_cluster else np.nan
    return theta_mean, theta_std, phi_mean, phi_std

def soft_kmeans_theta_phi_estimates(
        iteration,
        prior_theta_phi_estimates,
        data_we_have,
        biomarkers,
        order_dict,
        n_participants,
        non_diseased_participants,
        hashmap_of_normalized_stage_likelihood_dicts,
        diseased_stages,
        seed=None):
    """
    Get the DataFrame of theta and phi using the soft K-means algorithm for all biomarkers.

    Parameters:
        data_we_have (pd.DataFrame): DataFrame containing the data.
        biomarkers (list): List of biomarkers in string.
        order_dict (dict): Dictionary mapping biomarkers to their order.
        n_participants (int): Number of participants in the study.
        non_diseased_participants (list): List of non-diseased participants.
        hashmap_of_normalized_stage_likelihood_dicts (dict): Hash map of dictionaries containing stage likelihoods for each participant.
        diseased_stages (list): List of diseased stages.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        a dictionary containing the means and standard deviations for theta and phi for each biomarker.
    """
    # List of dicts to store the estimates
    # In each dic, key is biomarker, and values are theta and phi params
    hashmap_of_means_stds_estimate_dicts = {}
    for biomarker in biomarkers:
        dic = {'biomarker': biomarker}
        prior_theta_phi_estimates_biomarker = prior_theta_phi_estimates[biomarker]
        theta_mean, theta_std, phi_mean, phi_std = calculate_soft_kmeans_for_biomarker(
            data_we_have,
            biomarker,
            order_dict,
            n_participants,
            non_diseased_participants,
            hashmap_of_normalized_stage_likelihood_dicts,
            diseased_stages,
            seed
        )
        if theta_std == 0 or np.isnan(theta_std):
            theta_mean = prior_theta_phi_estimates_biomarker['theta_mean']
            theta_std = prior_theta_phi_estimates_biomarker['theta_std']
        if phi_std == 0 or np.isnan(phi_std):
            phi_mean = prior_theta_phi_estimates_biomarker['phi_mean']
            phi_std = prior_theta_phi_estimates_biomarker['phi_std']
        dic['theta_mean'] = theta_mean
        dic['theta_std'] = theta_std
        dic['phi_mean'] = phi_mean
        dic['phi_std'] = phi_std
        hashmap_of_means_stds_estimate_dicts[biomarker] = dic
    return hashmap_of_means_stds_estimate_dicts

def calculate_all_participant_ln_likelihood_and_update_hashmap(
        iteration,
        data_we_have,
        current_order_dict,
        n_participants,
        non_diseased_participant_ids,
        theta_phi_estimates,
        diseased_stages,
):
    data = data_we_have.copy()
    data['S_n'] = data.apply(
        lambda row: current_order_dict[row['biomarker']], axis=1)
    all_participant_ln_likelihood = 0
    # key is participant id
    # value is normalized_stage_likelihood_dict
    hashmap_of_normalized_stage_likelihood_dicts = {}
    for p in range(n_participants):
        pdata = data[data.participant == p].reset_index(drop=True)
        if p in non_diseased_participant_ids:
            this_participant_likelihood = utils.compute_likelihood(
                pdata, k_j=0, theta_phi=theta_phi_estimates)
            this_participant_ln_likelihood = np.log(
                this_participant_likelihood + 1e-10)
        else:
            # normalized_stage_likelihood_dict = None
            # initiaze stage_likelihood
            stage_likelihood_dict = {}
            for k_j in diseased_stages:
                kj_likelihood = utils.compute_likelihood(
                    pdata, k_j, theta_phi_estimates)
                # update each stage likelihood for this participant
                stage_likelihood_dict[k_j] = kj_likelihood
            # Add a small epsilon to avoid division by zero
            likelihood_sum = sum(stage_likelihood_dict.values())
            epsilon = 1e-10
            if likelihood_sum == 0:
                # print("Invalid likelihood_sum: zero encountered.")
                likelihood_sum = epsilon  # Handle the case accordingly
            normalized_stage_likelihood = [
                l/likelihood_sum for l in stage_likelihood_dict.values()]
            normalized_stage_likelihood_dict = dict(
                zip(diseased_stages, normalized_stage_likelihood))
            hashmap_of_normalized_stage_likelihood_dicts[p] = normalized_stage_likelihood_dict

            # calculate weighted average
            this_participant_likelihood = np.mean(likelihood_sum)
            this_participant_ln_likelihood = np.log(
                this_participant_likelihood)
        all_participant_ln_likelihood += this_participant_ln_likelihood
    return all_participant_ln_likelihood, hashmap_of_normalized_stage_likelihood_dicts


def metropolis_hastings_soft_kmeans(
    data_we_have,
    iterations,
    n_shuffle,
):
    '''Implement the metropolis-hastings algorithm using soft kmeans
    Inputs: 
        - data: data_we_have
        - iterations: number of iterations
        - log_folder_name: the folder where log files locate

    Outputs:
        - best_order: a numpy array
        - best_likelihood: a scalar 
    '''
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_biomarkers = len(biomarkers)
    n_stages = n_biomarkers + 1
    non_diseased_participant_ids = data_we_have.loc[
        data_we_have.diseased == False].participant.unique()
    diseased_stages = np.arange(start=1, stop=n_stages, step=1)
    # obtain the iniial theta and phi estimates
    prior_theta_phi_estimates = utils.get_theta_phi_estimates(
        data_we_have)
    theta_phi_estimates = prior_theta_phi_estimates.copy()

    # initialize empty lists
    acceptance_count = 0
    all_current_accepted_order_dicts = []

    current_accepted_order = np.random.permutation(np.arange(1, n_stages))
    current_accepted_order_dict = dict(zip(biomarkers, current_accepted_order))
    current_accepted_likelihood = -np.inf

    for _ in range(iterations):
        # in each iteration, we have updated current_order_dict and theta_phi_estimates

        new_order = current_accepted_order.copy()
        utils.shuffle_order(new_order, n_shuffle)
        current_order_dict = dict(zip(biomarkers, new_order))
        all_participant_ln_likelihood, \
            hashmap_of_normalized_stage_likelihood_dicts = calculate_all_participant_ln_likelihood_and_update_hashmap(
                _,
                data_we_have,
                current_order_dict,
                n_participants,
                non_diseased_participant_ids,
                theta_phi_estimates,
                diseased_stages,
            )

        # Now, update theta_phi_estimates using soft kmeans
        # based on the updated hashmap of normalized stage likelihood dicts
        theta_phi_estimates = soft_kmeans_theta_phi_estimates(
            _,
            prior_theta_phi_estimates,
            data_we_have,
            biomarkers,
            current_order_dict,
            n_participants,
            non_diseased_participant_ids,
            hashmap_of_normalized_stage_likelihood_dicts,
            diseased_stages,
            seed=None,
        )

        # Log-Sum-Exp Trick
        max_likelihood = max(all_participant_ln_likelihood,
                             current_accepted_likelihood)
        prob_of_accepting_new_order = np.exp(
            (all_participant_ln_likelihood - max_likelihood) -
            (current_accepted_likelihood - max_likelihood)
        )

        # prob_of_accepting_new_order = np.exp(
        #     all_participant_ln_likelihood - current_accepted_likelihood)

        # np.exp(a)/np.exp(b) = np.exp(a - b)
        # if a > b, then np.exp(a - b) > 1

        # it will definitly update at the first iteration
        if np.random.rand() < prob_of_accepting_new_order:
            acceptance_count += 1
            current_accepted_order = new_order
            current_accepted_likelihood = all_participant_ln_likelihood
            current_accepted_order_dict = current_order_dict

        acceptance_ratio = acceptance_count*100/(_+1)
        all_current_accepted_order_dicts.append(current_accepted_order_dict)

        if (_+1) % 10 == 0:
            formatted_string = (
                f"iteration {_ + 1} done, "
                f"current accepted likelihood: {current_accepted_likelihood}, "
                f"current acceptance ratio is {acceptance_ratio:.2f} %, "
                f"current accepted order is {current_accepted_order_dict.values()}, "
            )
            print(formatted_string)

    # print("done!")
    return all_current_accepted_order_dicts
