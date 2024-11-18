import json
import pandas as pd
from . import soft_kmeans_algo
from . import utils
from scipy.stats import kendalltau
import os
import logging
from typing import Optional, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_soft_kmeans(
    data_file: str,
    n_iter: int,
    n_shuffle: int,
    burn_in: int,
    thinning: int,
    heatmap_folder: str,
    filename: str,
    temp_result_file: str,
    dic: Optional[Dict[str, list]] = None,
) -> Dict[str, list]:
    """
    Run the soft kmeans algorithm and generate results.

    Args:
        data_file (str): Path to the input CSV file with biomarker data.
        n_iter (int): Number of iterations for the Metropolis-Hastings algorithm.
        n_shuffle (int): Number of shuffles per iteration.
        burn_in (int): Burn-in period for the MCMC chain.
        thinning (int): Thinning interval for the MCMC chain.
        heatmap_folder (str): Path to save the generated heatmaps.
        filename (str): Unique identifier for the current run.
        temp_result_file (str): Path to save the temporary results JSON.
        dic (Optional[Dict[str, list]]): Existing dictionary to store results.

    Returns:
        Dict[str, list]: Updated dictionary with Kendall's tau results.
    """
    if dic is None:
        dic = {}

    # Ensure directories exist
    os.makedirs(heatmap_folder, exist_ok=True)
    os.makedirs(os.path.dirname(temp_result_file), exist_ok=True)

    # Load data
    try:
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        logging.error(f"Data file not found: {data_file}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Data file is empty: {data_file}")
        raise
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        raise

    # Determine the number of biomarkers
    n_biomarkers = len(data.biomarker.unique())
    logging.info(f"Number of biomarkers: {n_biomarkers}")

    # Initialize results dictionary if not already present
    if filename not in dic:
        dic[filename] = []

    # Run the Metropolis-Hastings algorithm
    try:
        accepted_order_dicts = soft_kmeans_algo.metropolis_hastings_soft_kmeans(
            data, n_iter, n_shuffle
        )
    except Exception as e:
        logging.error(f"Error in Metropolis-Hastings algorithm: {e}")
        raise

    # Save heatmap
    try:
        utils.save_heatmap(
            accepted_order_dicts,
            burn_in,
            thinning,
            folder_name=heatmap_folder,
            file_name=filename,
            title=f"Heatmap of {filename}",
        )
    except Exception as e:
        logging.error(f"Error generating heatmap: {e}")
        raise

    # Calculate the most likely order
    try:
        most_likely_order_dic = utils.obtain_most_likely_order_dic(
            accepted_order_dicts, burn_in, thinning
        )
        most_likely_order = list(most_likely_order_dic.values())
        tau, p_value = kendalltau(most_likely_order, range(1, n_biomarkers + 1))
    except Exception as e:
        logging.error(f"Error calculating Kendall's tau: {e}")
        raise

    # Store results
    dic[filename].append(tau)
    logging.info(f"Kendall's tau for {filename}: {tau}")

    # Save results to file
    try:
        with open(temp_result_file, "w") as file:
            json.dump(dic, file, indent=4)
        logging.info(f"Results saved to {temp_result_file}")
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")
        raise

    return dic
