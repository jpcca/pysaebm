import json
import pandas as pd
import os
import logging
from typing import List, Dict
from scipy.stats import kendalltau
import re 

from alabEBM.algorithms.soft_kmeans_algo import metropolis_hastings_soft_kmeans
from alabEBM.utils.visualization import save_heatmap, save_traceplot 
from alabEBM.utils.logging_utils import setup_logging 
from alabEBM.utils.data_processing import get_theta_phi_estimates, obtain_most_likely_order_dic
from alabEBM.utils.runners import extract_fname, cleanup_old_files

def run_soft_kmeans(
    data_file: str,
    n_iter: int = 2000,
    n_shuffle: int = 2,
    burn_in: int = 1000,
    thinning: int = 50,
    output_dir : str = "sk",
) -> Dict[str, float]:
    """
    Run the soft k-means algorithm and save results in the `sk` folder.

    Args:
        data_file (str): Path to the input CSV file with biomarker data.
        fname (str): 
        n_iter (int): Number of iterations for the Metropolis-Hastings algorithm.
        n_shuffle (int): Number of shuffles per iteration.
        burn_in (int): Burn-in period for the MCMC chain.
        thinning (int): Thinning interval for the MCMC chain.
        output_dir (str): Folder to save all outputs (heatmaps, traceplots, results, logs).

    Returns:
        Dict[str, float]: Results including Kendall's tau and p-value.
    """
    fname = extract_fname(data_file)
    # Clean up old files before running
    cleanup_old_files(output_dir, fname)

    # Create output direcories 
    os.makedirs(output_dir, exist_ok = True)
    heatmap_folder = f"{output_dir}/heatmaps"
    traceplot_folder = f"{output_dir}/traceplots"
    results_folder = f"{output_dir}/results"
    logs_folder = f"{output_dir}/logs"

    os.makedirs(heatmap_folder, exist_ok=True)
    os.makedirs(traceplot_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    # Set up logging 
    log_file = f"{logs_folder}/{fname}.log"
    setup_logging(log_file)

    # Log the start of the run
    logging.info(f"Running soft k-means for file: {fname}")

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
        accepted_order_dicts, log_likelihoods = metropolis_hastings_soft_kmeans(
            data, n_iter, n_shuffle
        )
    except Exception as e:
        logging.error(f"Error in Metropolis-Hastings algorithm: {e}")
        raise

    # Save heatmap
    try:
        save_heatmap(
            accepted_order_dicts,
            burn_in,
            thinning,
            folder_name=heatmap_folder,
            file_name=f"{fname}_heatmap",
            title=f"Heatmap of {fname}",
        )
    except Exception as e:
        logging.error(f"Error generating heatmap: {e}")
        raise

    # Save trace plot
    try:
        save_traceplot(log_likelihoods, traceplot_folder, f"{fname}_traceplot")
    except Exception as e:
        logging.error(f"Error generating trace plot: {e}")
        raise 

    # Calculate the most likely order
    try:
        most_likely_order_dic = obtain_most_likely_order_dic(
            accepted_order_dicts, burn_in, thinning
        )
        most_likely_order = list(most_likely_order_dic.values())
        tau, p_value = kendalltau(most_likely_order, range(1, n_biomarkers + 1))
    except Exception as e:
        logging.error(f"Error calculating Kendall's tau: {e}")
        raise

    # Save results 
    results = {
        "most_likely_order": most_likely_order_dic,
        "kendalls_tau": tau, 
        "p_value": p_value,
    }
    try:
        with open(f"{results_folder}/{fname}_results.json", "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")
        raise 
    logging.info(f"Results saved to {results_folder}/{fname}_results.json")

    return results 
