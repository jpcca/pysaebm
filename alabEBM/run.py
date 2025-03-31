import json
import pandas as pd
import os
import logging
from typing import List, Dict, Optional
from scipy.stats import kendalltau
import re 

# Import utility functions
from alabebm.utils.visualization import save_heatmap, save_traceplot 
from alabebm.utils.logging_utils import setup_logging 
from alabebm.utils.data_processing import get_theta_phi_estimates, obtain_most_likely_order_dic
from alabebm.utils.runners import extract_fname, cleanup_old_files

# Import algorithms
from alabebm.algorithms import metropolis_hastings

def run_ebm(
    data_file: str,
    algorithm: str, 
    n_iter: int = 2000,
    n_shuffle: int = 2,
    burn_in: int = 1000,
    thinning: int = 50,
    correct_ordering: Optional[Dict[str, int]] = None,
    plot_title_detail: Optional[str] = "",
    fname_prefix: Optional[str] = "",
    skip_heatmap: Optional[bool] = False,
    skip_traceplot: Optional[bool] = False,
) -> Dict[str, float]:
    """
    Run the metropolis hastings algorithm and save results 

    Args:
        data_file (str): Path to the input CSV file with biomarker data.
        algorithm (str): Choose from 'hard_kmeans', 'mle', and 'conjugate_priors'.
        n_iter (int): Number of iterations for the Metropolis-Hastings algorithm.
        n_shuffle (int): Number of shuffles per iteration.
        burn_in (int): Burn-in period for the MCMC chain.
        thinning (int): Thinning interval for the MCMC chain.
        correct_ordering (Optional[Dict[str, int]]): biomarker name: the initial correct order of it (if known)
        plot_title_detail (Optional[str]): optional string to add to plot title. 
        fname_prefix (Optional[str]): the prefix of heatmap, traceplot, results.json, and logs file, e.g., 5_50_0_heatmap_conjugate_priors.png
            In the example, there are no prefix strings. 
        skip_heatmap (Optional[bool]): whether to save heatmaps. True if want to save space.
        skip_traceplot (Optional[bool]): whether to save traceplots. True if want to save space.

    Returns:
        Dict[str, float]: Results including Kendall's tau and p-value.
    """
    allowed_algorithms = {'hard_kmeans', 'mle', 'conjugate_priors', 'em'}  # Using a set for faster lookup
    if algorithm not in allowed_algorithms:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Must be one of {allowed_algorithms}")

    # Folder to save all outputs
    output_dir = algorithm
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

    os.makedirs(heatmap_folder, exist_ok=True)
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

    # Run the Metropolis-Hastings algorithm
    try:
        accepted_order_dicts, log_likelihoods = metropolis_hastings(data, n_iter, n_shuffle, algorithm)
    except Exception as e:
        logging.error(f"Error in Metropolis-Hastings algorithm: {e}")
        raise

    # Get the order associated with the highet log likelihoods
    order_with_higest_ll = accepted_order_dicts[log_likelihoods.index(max(log_likelihoods))]
    # Sort by keys in an ascending order
    order_with_higest_ll = dict(sorted(order_with_higest_ll.items()))
    if correct_ordering:
        # Sort both dicts by the key to make sure they are comparable
        correct_ordering = dict(sorted(correct_ordering.items()))
        tau2, p_value2 = kendalltau(
            list(order_with_higest_ll.values()), 
            list(correct_ordering.values()))
    else:
        tau2, p_value2, original_order = None, None, None 

    # Calculate the most likely order
    try:
        most_likely_order_dic = obtain_most_likely_order_dic(
            accepted_order_dicts, burn_in, thinning
        )
        most_likely_order_dic = dict(sorted(most_likely_order_dic.items()))
        # most_likely_order = list(most_likely_order_dic.values())
        # Only calculate tau and p_value if correct_ordering is provided
        if correct_ordering:
            tau, p_value = kendalltau(
                list(most_likely_order_dic.values()), 
                list(correct_ordering.values()))
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

    if correct_ordering:
        original_order = dict(sorted(correct_ordering.items(), key=lambda item:item[1]))
    else:
        original_order = correct_ordering
    # Save results 
    results = {
        "n_iter": n_iter,
        "n_shuffle": n_shuffle, 
        "burn_in": burn_in,
        "thinning": thinning,
        "most_likely_order": dict(sorted(most_likely_order_dic.items(), key=lambda item:item[1])),
        "kendalls_tau": tau, 
        "p_value": p_value,
        "original_order": original_order,
        "order_with_higest_ll": {k: int(v) for k, v in sorted(order_with_higest_ll.items(), key=lambda item: item[1])},
        "kendalls_tau2": tau2,
        "p_value2": p_value2
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
