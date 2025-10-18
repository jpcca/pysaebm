import json
import pandas as pd
import os
import logging
from typing import List, Dict, Optional, Union
from scipy.stats import kendalltau
import time
import numpy as np
from sklearn.metrics import mean_absolute_error

# Import utility functions
from .utils import (setup_logging, 
                   extract_fname, 
                   cleanup_old_files, 
                   stage_with_plugin_pi_em
                   )
from .viz import save_heatmap, save_traceplot

# Import algorithms
from .mh import metropolis_hastings

from .kde_mh import metropolis_hastings_kde

from .kde_utils import (
    preprocess_participant_data,
    stage_with_plugin_pi_em_kde
)

def run_ebm(
    algorithm:str,
    data_file: str,
    output_dir: str,
    output_folder: Optional[str] = None,
    n_iter: int = 2000,
    n_shuffle: int = 2,
    burn_in: int = 500,
    thinning: int = 1,
    true_order_dict: Optional[Dict[str, int]] = None,
    true_stages: Optional[List[int]] = None,
    plot_title_detail: Optional[str] = "",
    fname_prefix: Optional[str] = "",
    skip_heatmap: Optional[bool] = False,
    skip_traceplot: Optional[bool] = False,
    # Strength of the prior belief in prior estimate of the mean (μ), set to 1 as default
    prior_n: float = 1.0,
    # Prior degrees of freedom, influencing the certainty of prior estimate of the variance (σ²), set to 1 as default
    prior_v: float = 1.0,
    seed: int = 123,
    save_results:bool=True,
    save_theta_phi:bool=False,
    save_stage_post:bool=False,
    save_details:bool=False,
    max_iter_staging:Optional[int]=100,
) -> Dict[str, Union[str, int, float, Dict, List]]:
    """
    Run the metropolis hastings algorithm and save results 

    Args:
        algorithm (str): Choose from 'hard_kmeans', 'mle', 'em', 'kde', and 'conjugate_priors' (default).
        data_file (str): Path to the input CSV file with biomarker data.
        output_dir (str): Path to the directory to store all the results.
        output_folder (str): Optional. If not provided, all results will be saved to output_dir/algorithm. 
            If provided, results will be saved to output_dir/output_folder
        n_iter (int): Number of iterations for the Metropolis-Hastings algorithm.
        n_shuffle (int): Number of shuffles per iteration.
        burn_in (int): Burn-in period for the MCMC chain.
        thinning (int): Thinning interval for the MCMC chain.
        true_order_dict (Optional[Dict[str, int]]): biomarker name: the correct order of it (if known)
        true_stages (Optional[List[int]]): true stages for all participants (if known)
        plot_title_detail (Optional[str]): optional string to add to plot title, as suffix.
        fname_prefix (Optional[str]): the prefix of heatmap, traceplot, results.json, and logs file, e.g., 5_50_0_heatmap_conjugate_priors.png
            In the example, there are no prefix strings. 
        skip_heatmap (Optional[bool]): whether to save heatmaps. True you want to skip saving heatmaps and save space.
        skip_traceplot (Optional[bool]): whether to save traceplots. True if you want to skip saving traceplots and save space.
        prior_n (strength of belief in prior of mean): default to be 1.0
        prior_v (prior degree of freedom) are the weakly informative priors, default to be 1.0
        seed (int): for reproducibility
        save_results: whether to save the json result.
        save_theta_phi: if save_results, whether to include the theta_phi result.
        save_stage_post: if save_results, whether to include the stage post result. 
        save_details: if save_results, save the simple version instead of the complete version. 

    Returns:
        Dict[str, Union[str, int, float, Dict, List]]: Results including everything, e.g., Kendall's tau and p-value.

        Whether to save results or not, the results will be returned. If save_details, the complete veresion will be returned.
        Otherwise, the simple version will be returned. 
    """
    start_time = time.time()

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Using a set for faster lookup
    allowed_algorithms = {'hard_kmeans', 'mle', 'conjugate_priors', 'em', 'kde'}
    if algorithm not in allowed_algorithms:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Must be one of {allowed_algorithms}")

    # Folder to save all outputs
    if output_folder:
        output_dir = os.path.join(output_dir, output_folder)
    else:
        output_dir = os.path.join(output_dir, algorithm)
    fname = extract_fname(data_file)

    # First do cleanup
    logging.info(f"Starting cleanup for {algorithm.replace('_', ' ')}...")
    cleanup_old_files(output_dir, fname)

    # Then create directories
    os.makedirs(output_dir, exist_ok=True)

    heatmap_folder = os.path.join(output_dir, "heatmaps")
    traceplot_folder = os.path.join(output_dir, "traceplots")
    results_folder = os.path.join(output_dir, "results")
    logs_folder = os.path.join(output_dir, "records")

    if not skip_heatmap:
        os.makedirs(heatmap_folder, exist_ok=True)
    if not skip_traceplot:
        os.makedirs(traceplot_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    # Finally set up logging
    log_file = os.path.join(logs_folder, f"{fname_prefix}{fname}.log")
    setup_logging(log_file)

    # Log the start of the run
    logging.info(f"Running {algorithm.replace('_', ' ')} for file: {fname}")
    logging.getLogger().handlers[0].flush()  # Flush logs immediately

    # Load data
    try:
        data = pd.read_csv(data_file)
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        raise

    # sort biomarkeres by name, ascending
    biomarker_names = sorted(data.biomarker.unique())
    n_biomarkers = len(biomarker_names)
    # n_stages = n_biomarkers + 1
    logging.info(f"Number of biomarkers: {n_biomarkers}")

    n_participants = len(data.participant.unique())

    df = data.copy()
    diseased_dict = dict(zip(df.participant, df.diseased))
    dff = df.pivot(
        index='participant', columns='biomarker', values='measurement')
    # make sure the data_matrix is in this order
    dff = dff.reindex(columns=biomarker_names, level=1) 
    # remove column name (biomarker) to clean display
    dff.columns.name = None      
    # bring 'participant' back as a column and then delete it
    dff.reset_index(inplace=True, drop=True)  
    data_matrix = dff.to_numpy()
    diseased_arr = np.array([int(diseased_dict[x]) for x in range(n_participants)])

    non_diseased_ids = np.where(diseased_arr == 0)[0]
    healthy_ratio = len(non_diseased_ids)/n_participants

    if algorithm == 'kde':
        # Run the Metropolis-Hastings algorithm
        try:
            accepted_orders, log_likelihoods, best_order, best_theta_phi, best_stage_prior = metropolis_hastings_kde(
                data_we_have=data, iterations=n_iter, n_shuffle=n_shuffle, rng=rng, burn_in=burn_in
            )
        except Exception as e:
            logging.error(f"Error in Metropolis-Hastings KDE algorithm: {e}")
            raise

    else:
        # Run the Metropolis-Hastings algorithm
        try:
            accepted_orders, log_likelihoods, best_order, best_theta_phi, best_stage_prior = metropolis_hastings(
                algorithm=algorithm, data_matrix=data_matrix, diseased_arr=diseased_arr, iterations = n_iter, 
                n_shuffle = n_shuffle,  prior_n=prior_n, prior_v=prior_v, rng=rng, burn_in=burn_in
            )
            
        except Exception as e:
            logging.error(f"Error in Metropolis-Hastings algorithm: {e}")
            raise

    # Get the order associated with the highet log likelihoods
    order_with_highest_ll = best_order # best order starts from 1!!! both for mh.py and kde_mh.py
    
    if true_order_dict:
        # Sort both dicts by the key to make sure they are comparable
        true_order_dict = dict(sorted(true_order_dict.items()))
        true_order_indices = np.array(list(true_order_dict.values())) 
        print('order with highest ll: ', order_with_highest_ll)
        print('true order indices: ', true_order_indices)
        tau, p_value = kendalltau(order_with_highest_ll, true_order_indices)
        tau = (1-tau)/2
    else:
        tau, p_value = None, None

    pretty_algo_name_dict = {
        'conjugate_priors': 'Conjugate Priors',
        'hard_kmeans': 'Hard K-Means',
        'kde': 'KDE',
        'mle': 'MLE',
        'em': 'EM'
    }

    try:
        pretty_name = pretty_algo_name_dict[algorithm]
    except:
        pretty_name = algorithm.replace("_", " ").title()

    # Save heatmap
    if save_results and not skip_heatmap:
        try:
            save_heatmap(
                accepted_orders,
                burn_in,
                thinning,
                folder_name=heatmap_folder,
                file_name=f"{fname_prefix}{fname}_heatmap_{algorithm}",
                title=f"{pretty_name} Ordering Result {plot_title_detail}",
                biomarker_names=biomarker_names,
                best_order=order_with_highest_ll
            )
        except Exception as e:
            logging.error(f"Error generating heatmap: {e}")
            raise

    # Save trace plot
    if save_results and not skip_traceplot:
        try:
            save_traceplot(
                log_likelihoods,
                folder_name=traceplot_folder,
                file_name=f"{fname_prefix}{fname}_traceplot_{algorithm}",
                title=f"Traceplot of Log Likelihoods ({pretty_name}) {plot_title_detail}"
            )
        except Exception as e:
            logging.error(f"Error generating trace plot: {e}")
            raise
    
    if algorithm == 'kde':
        order_with_highest_ll_dict = dict(zip(biomarker_names, order_with_highest_ll))
        participant_data = preprocess_participant_data(data, order_with_highest_ll_dict)

        final_stage_post, ml_stages, updated_pi = stage_with_plugin_pi_em_kde(
            participant_data=participant_data,
            final_theta_phi=best_theta_phi,
            rng=rng,
            max_iter=max_iter_staging,
            tol=1e-6
        )
        
    else:
        final_stage_post, ml_stages, updated_pi = stage_with_plugin_pi_em(
            data_matrix=data_matrix,
            order_with_highest_ll=order_with_highest_ll,  # no need to + 1 because in pysaebm, the order starts from 1 already. 
            # this in fact is okay, because current_order, the ordering index always starts from 1. 
            # in inference with label, the disease stages, or stage_post, has n_biomarker diseases starting from 1
            # but in blind inference, we have n_stages, starting from 0. 
            final_theta_phi=best_theta_phi,
            rng=rng,
            tol=1e-6,
            max_iter=max_iter_staging
        )

    mae = None
    true_order_result = None

    if true_stages:
        mae = mean_absolute_error(true_stages, ml_stages)
    if true_order_dict:
        true_order_result = {k: int(v) for k, v in true_order_dict.items()}
    
    final_stage_post_dict = {}
    final_theta_phi_dict = {}
    if save_results:
        if save_stage_post:
            for p in range(n_participants):
                final_stage_post_dict[p] = final_stage_post[p].astype(float).tolist()
        
        if save_theta_phi:
            if algorithm =='kde':
                final_theta_phi_dict = best_theta_phi
            else:
                if algorithm != 'kde':
                    for bm_idx, bm in enumerate(biomarker_names):
                        params = best_theta_phi[bm_idx]
                        final_theta_phi_dict[bm] = {
                            'theta_mean': params[0],
                            'theta_std': params[1],
                            'phi_mean': params[2],
                            'phi_std': params[3]
                        }

    end_time = time.time()
    if save_details:
        results = {
            "algorithm": algorithm,
            "runtime": end_time - start_time,
            "N_MCMC": n_iter,
            "n_shuffle": n_shuffle,
            "burn_in": burn_in,
            "thinning": thinning,
            'healthy_ratio': healthy_ratio,
            "max_log_likelihood": float(max(log_likelihoods)),
            "kendalls_tau": tau,
            "p_value": p_value,
            "mean_absolute_error": mae,
            # the stage prior associated with the best_order
            'current_pi': best_stage_prior.astype(float).tolist(),
            # updated pi is the pi for all stages, including 0
            'updated_pi': updated_pi.astype(float).tolist(),
            'true_order': true_order_result,
            "order_with_highest_ll": {k: int(v) for k, v in zip(biomarker_names, order_with_highest_ll)},
            "true_stages": true_stages,
            'ml_stages': ml_stages.astype(int).tolist(),
            "stage_likelihood_posterior": final_stage_post_dict,
            "final_theta_phi_params": final_theta_phi_dict,
        }
    else:
        results = {
            "algorithm": algorithm,
            "runtime": end_time - start_time,
            'healthy_ratio': healthy_ratio,
            "max_log_likelihood": float(max(log_likelihoods)),
            "kendalls_tau": tau,
            "mean_absolute_error": mae,
        }
    if save_results:
    # Save results
        try:
            with open(f"{results_folder}/{fname_prefix}{fname}_results.json", "w") as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            logging.error(f"Error writing results to file: {e}")
            raise
        logging.info(f"Results saved to {results_folder}/{fname_prefix}{fname}_results.json")

    # Clean up logging handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    return results