import json
import pandas as pd
import os
import logging
from typing import List, Dict, Optional, Union
from scipy.stats import kendalltau
import re 
import math 
import numpy as np 

# Import utility functions
from pysaebm.utils.visualization import save_heatmap, save_traceplot 
from pysaebm.utils.logging_utils import setup_logging 
from pysaebm.utils.data_processing import obtain_most_likely_order_dic
from pysaebm.utils.runners import extract_fname, cleanup_old_files

# Import algorithms
from pysaebm.algorithms import metropolis_hastings
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error

def run_ebm(
    data_file: str,
    output_dir: str,
    algorithm: str = 'conjugate_priors', 
    n_iter: int = 2000,
    n_shuffle: int = 2,
    burn_in: int = 500,
    thinning: int = 10,
    true_order_dict: Optional[Dict[str, int]] = None,
    true_stages: Optional[List[int]] = None,
    plot_title_detail: Optional[str] = "",
    fname_prefix: Optional[str] = "",
    skip_heatmap: Optional[bool] = False,
    skip_traceplot: Optional[bool] = False,
    prior_n: float = 1.0,    # Strength of the prior belief in prior estimate of the mean (μ), set to 1 as default
    prior_v: float = 1.0,     # Prior degrees of freedom, influencing the certainty of prior estimate of the variance (σ²), set to 1 as default
    weight_change_threshold: float = 0.01,
    bw_method: str = 'scott'
) -> Dict[str, Union[str, int, float, Dict, List]]:
    """
    Run the metropolis hastings algorithm and save results 

    Args:
        data_file (str): Path to the input CSV file with biomarker data.
        output_dir (str): Path to the directory to store all the results.
        algorithm (str): Choose from 'hard_kmeans', 'mle', 'em', 'kde', and 'conjugate_priors' (default).
        n_iter (int): Number of iterations for the Metropolis-Hastings algorithm.
        n_shuffle (int): Number of shuffles per iteration.
        burn_in (int): Burn-in period for the MCMC chain.
        thinning (int): Thinning interval for the MCMC chain.
        true_order_dict (Optional[Dict[str, int]]): biomarker name: the correct order of it (if known)
        true_stages (Optional[List[int]]): true stages for all participants (if known)
        plot_title_detail (Optional[str]): optional string to add to plot title. 
        fname_prefix (Optional[str]): the prefix of heatmap, traceplot, results.json, and logs file, e.g., 5_50_0_heatmap_conjugate_priors.png
            In the example, there are no prefix strings. 
        skip_heatmap (Optional[bool]): whether to save heatmaps. True if want to skip saving heatmpas and save space.
        skip_traceplot (Optional[bool]): whether to save traceplots. True if want to skip saving traceplots and save space.
        prior_n (strength of belief in prior of mean): default to be 1.0
        prior_v (prior degree of freedom) are the weakly informative priors, default to be 1.0
        weight_change_threshold (float): Threshold for kde weights (if np.mean(new_weights - old_weights)) > threshold, then recalculate
            otherwise use the new kde and weights
        bw_method (str): bandwidth selection method in kde

    Returns:
        Dict[str, Union[str, int, float, Dict, List]]: Results including everything, e.g., Kendall's tau and p-value.
    """
    allowed_algorithms = {'hard_kmeans', 'mle', 'conjugate_priors', 'em', 'kde', 'gmm', 'gmm_dm_prior', 'conjugate_priors_dm_prior'}  # Using a set for faster lookup
    if algorithm not in allowed_algorithms:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Must be one of {allowed_algorithms}")

    # Folder to save all outputs
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

    # Determine the number of biomarkers
    n_biomarkers = len(data.biomarker.unique())
    logging.info(f"Number of biomarkers: {n_biomarkers}")

    # Determine the number of biomarkers
    n_participants = len(data.participant.unique())

    # Run the Metropolis-Hastings algorithm
    try:
        accepted_order_dicts, log_likelihoods, final_theta_phi_params, final_stage_post, current_pi = metropolis_hastings(
            data, n_iter, n_shuffle, algorithm, prior_n=prior_n, prior_v=prior_v, weight_change_threshold = weight_change_threshold, bw_method=bw_method)
    except Exception as e:
        logging.error(f"Error in Metropolis-Hastings algorithm: {e}")
        raise

    # Get the order associated with the highet log likelihoods
    order_with_higest_ll = accepted_order_dicts[log_likelihoods.index(max(log_likelihoods))]
    # Sort by keys in an ascending order
    order_with_higest_ll = dict(sorted(order_with_higest_ll.items()))
    if true_order_dict:
        # Sort both dicts by the key to make sure they are comparable
        true_order_dict = dict(sorted(true_order_dict.items()))
        tau2, p_value2 = kendalltau(
            list(order_with_higest_ll.values()), 
            list(true_order_dict.values()))
    else:
        tau2, p_value2 = None, None

    # Calculate the most likely order
    try:
        most_likely_order_dic = obtain_most_likely_order_dic(
            accepted_order_dicts, burn_in, thinning
        )
        most_likely_order_dic = dict(sorted(most_likely_order_dic.items()))
        # most_likely_order = list(most_likely_order_dic.values())
        # Only calculate tau and p_value if true_order_dict is provided
        if true_order_dict:
            tau, p_value = kendalltau(
                list(most_likely_order_dic.values()), 
                list(true_order_dict.values()))
        else:
            tau, p_value = None, None
    except Exception as e:
        logging.error(f"Error calculating Kendall's tau: {e}")
        raise

    # Save heatmap
    if not skip_heatmap:
        try:
            save_heatmap(
                accepted_order_dicts,
                burn_in,
                thinning,
                folder_name=heatmap_folder,
                file_name=f"{fname_prefix}{fname}_heatmap_{algorithm}",
                title=f"Heatmap of {fname_prefix}{fname} Using {algorithm}, {plot_title_detail}",
                best_order = most_likely_order_dic
            )
        except Exception as e:
            logging.error(f"Error generating heatmap: {e}")
            raise

    # Save trace plot
    if not skip_traceplot:
        try:
            save_traceplot(
                log_likelihoods, 
                folder_name = traceplot_folder, 
                file_name = f"{fname_prefix}{fname}_traceplot_{algorithm}",
                title = f"Traceplot of Log Likelihoods" 
            )
        except Exception as e:
            logging.error(f"Error generating trace plot: {e}")
            raise 
    
    if algorithm == 'kde':
        final_theta_phi_params = {
            str(k): {kk: vv.tolist() for kk, vv in v.items()}
            for k, v in final_theta_phi_params.items()
        }

    # Most likely stage for all participants
    ml_stages = [
        np.random.choice(len(final_stage_post[pid]), p=final_stage_post[pid]) + 1
        if pid in final_stage_post else 0 for pid in range(n_participants)
    ]
    
    qwk = None 
    mae = None
    mse = None
    rmse = None

    if true_stages:
        qwk = cohen_kappa_score(true_stages, ml_stages, weights='quadratic')
        mae = mean_absolute_error(true_stages, ml_stages)
        mse = mean_squared_error(true_stages, ml_stages)
        rmse = math.sqrt(mse)

    # Save results 
    results = {
        "algorithm": algorithm,
        "N_MCMC": n_iter,
        "n_shuffle": n_shuffle, 
        "burn_in": burn_in,
        "thinning": thinning,
        "kendalls_tau2": tau2,
        "p_value2": p_value2,
        "kendalls_tau": tau, 
        "p_value": p_value,
        "quadratic_weighted_kappa": qwk,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        'current_pi': current_pi.tolist(),
        'true_order': {k:int(v) for k, v in true_order_dict.items()},
        'ml_order': {k:int(v) for k, v in most_likely_order_dic.items()},
        "order_with_higest_ll": {k: int(v) for k, v in order_with_higest_ll.items()},
        "true_stages": true_stages,
        'ml_stages': ml_stages,
        "stage_likelihood_posterior": {str(k): v.tolist() for k, v in final_stage_post.items()},
        "final_theta_phi_params": final_theta_phi_params,
    }
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
