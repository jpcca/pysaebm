import numpy as np
from . import utils

def calculate_all_participant_ln_likelihood(
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

            # calculate weighted average
            this_participant_likelihood = np.mean(likelihood_sum)
            this_participant_ln_likelihood = np.log(
                this_participant_likelihood + 1e-10)
        all_participant_ln_likelihood += this_participant_ln_likelihood
    return all_participant_ln_likelihood

def metropolis_hastings_hard_kmeans(
    data_we_have,
    iterations,
    n_shuffle,
):
    '''Implement the metropolis-hastings algorithm using simple clustering
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
    theta_phi_estimates = utils.get_theta_phi_estimates(
        data_we_have)

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
        all_participant_ln_likelihood = calculate_all_participant_ln_likelihood(
                _,
                data_we_have,
                current_order_dict,
                n_participants,
                non_diseased_participant_ids,
                theta_phi_estimates,
                diseased_stages,
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