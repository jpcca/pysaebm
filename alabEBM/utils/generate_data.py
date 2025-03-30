from typing import List, Optional, Tuple, Dict
import json 
import pandas as pd 
import numpy as np 
import os 
import scipy.stats as stats
from collections import Counter 

def generate_data_from_ebm(
    n_participants: int,
    biomarker_order: Dict[str, int],
    real_theta_phi_file: str,
    healthy_ratio: float,
    output_dir: str,
    m: int,  # combstr_m
    seed: int,
    stage_distribution: Optional[str] = 'dirichlet',
    prefix: Optional[str] = None,  # Optional prefix
    suffix: Optional[str] = None,   # Optional suffix,
    keep_all_cols: Optional[bool] = False 
) -> pd.DataFrame:
    """
    Simulate an Event-Based Model (EBM) for disease progression.

    Args:
    n_participants (int): Number of participants.
    biomarker_order (Dict[str, int]): Biomarker names and their orders
        in which each of them get affected by the disease.
    real_theta_phi_file (str): Directory of a JSON file which contains 
        theta and phi values for all biomarkers.
        See real_theta_phi.json for example format.
    output_dir (str): Directory where output files will be saved.
    healthy_ratio (float): Proportion of healthy participants out of n_participants.
    seed (int): Seed for the random number generator for reproducibility.
    stage_distribution (Optional[str]), either 'uniform' or 'dirichlet'
    prefix (Optional[str]): Optional prefix of filename
    suffix (Optional[str]): Optional suffix of filename
    keep_all_cols (Optional[bool]): if False, drop ['k_j', 'S_n', 'affected_or_not']

    Returns:
    pd.DataFrame: A DataFrame with columns 'participant', "biomarker", 'measurement', 
        'diseased', with or without ['k_j', 'S_n', 'affected_or_not']
    """
    # Parameter validation
    assert n_participants > 0, "Number of participants must be greater than 0."
    assert 0 <= healthy_ratio <= 1, "Healthy ratio must be between 0 and 1."
    assert stage_distribution in ['dirichlet', 'uniform'], "stage_distribution must be either dirichlet or uniform!"

    # Change dict to list 
    biomarker_order = dict(sorted(biomarker_order.items(), key=lambda item:item[1]))
    ordered_biomarkers = list(biomarker_order.keys())

    # Set the seed for numpy's random number generator
    rng = np.random.default_rng(seed)

    # Load theta and phi values from the JSON file
    try:
        with open(real_theta_phi_file) as f:
            real_theta_phi = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {real_theta_phi_file} not found")
    except json.JSONDecodeError:
        raise ValueError(
            f"File {real_theta_phi_file} is not a valid JSON file.")

    n_biomarkers = len(ordered_biomarkers)
    n_stages = n_biomarkers + 1
    disease_stages = np.arange(1, n_biomarkers + 1)
    n_diseased_stages = n_biomarkers

    n_healthy = int(n_participants * healthy_ratio)
    n_diseased = int(n_participants - n_healthy)

    if n_diseased > 0:
        if stage_distribution == 'dirichlet':
            # this output would be an array (of length n_biomarkers) of floats and this array sums up 1
            # note that some of the probs in the array might be 0
            stage_probs = rng.dirichlet(np.ones(n_biomarkers))
        elif stage_distribution == 'uniform':
            stage_probs = np.ones(n_biomarkers)/n_biomarkers
        # this output would be an array of stages
        # for example, counts = [3, 7, 0] which means 3 people in stage1, 7 in stage2, and 0 in stage3
        # and the sum of this array would be n_diseased
        counts = rng.multinomial(n_diseased, stage_probs)
        diseased_stages = np.repeat(disease_stages, counts)
    else:
        diseased_stages = np.array([], dtype=int)

    # Generate disease stages
    kjs = np.concatenate([np.zeros(n_healthy, dtype=int), diseased_stages])
    # shuffle so that it's not 0s first and then disease stages but all random
    rng.shuffle(kjs)

    print(Counter(kjs))

    # Initiate biomarker measurement matrix (J participants x N biomarkers) with None
    # X = np.full((n_participants, n_biomarkers), None, dtype=object)
    X = np.empty((n_participants, n_biomarkers), dtype=object)

    # Create distributions for each biomarker
    theta_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['theta_mean'],
        real_theta_phi[biomarker]['theta_std']
    ) for biomarker in ordered_biomarkers}

    phi_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['phi_mean'],
        real_theta_phi[biomarker]['phi_std']
    ) for biomarker in ordered_biomarkers}

    # Populate the matrix with biomarker measurements
    for j in range(n_participants):
        for n, biomarker in enumerate(ordered_biomarkers):
            # because for each j, we generate X[j, n] in the order of ordered_biomarkers,
            # the final dataset will have this ordering as well.
            k_j = kjs[j]
            S_n = n + 1

            # Assign biomarker values based on the participant's disease stage
            # affected, or not_affected, is regarding the biomarker, not the participant
            if k_j >= 1:
                if k_j >= S_n:
                    # rvs() is affected by np.random()
                    X[j, n] = (
                        j, biomarker, theta_dist[biomarker].rvs(random_state=rng), k_j, S_n, 'affected')
                else:
                    X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
                               k_j, S_n, 'not_affected')
            # if the participant is healthy
            else:
                X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
                           k_j, S_n, 'not_affected')

    df = pd.DataFrame(X, columns=ordered_biomarkers)
    # make this dataframe wide to long
    df_long = df.melt(var_name="Biomarker", value_name="Value")
    data = df_long['Value'].apply(pd.Series)
    data.columns = ['participant', "biomarker",
                    'measurement', 'k_j', 'S_n', 'affected_or_not']

    # biomarker_name_change_dic = dict(
    #     zip(ordered_biomarkers, range(1, n_biomarkers + 1)))
    data['diseased'] = data.apply(lambda row: row.k_j > 0, axis=1)
    if not keep_all_cols:
        data.drop(['k_j', 'S_n', 'affected_or_not'], axis=1, inplace=True)
    # data['biomarker'] = data.apply(
    #     lambda row: f"{row.biomarker} ({biomarker_name_change_dic[row.biomarker]})", axis=1)

    filename = f"{int(healthy_ratio*n_participants)}_{n_participants}_{m}"
    if prefix:
        filename = f"{prefix}_{filename}"
    if suffix:
        filename = f"{filename}_{suffix}"
    
    output_path = os.path.join(output_dir, f"{filename}.csv")
    data.to_csv(output_path, index=False)
    print("Data generation done! Output saved to:", filename)
    return data

def generate(
    biomarker_order: Dict[str, int],
    real_theta_phi_file: str,
    js: List[int],
    rs: List[float],
    num_of_datasets_per_combination: int,
    output_dir: str = 'data',
    seed: Optional[int] = None,
    stage_distribution: Optional[str] = 'dirichlet',
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    keep_all_cols: Optional[bool] = False 
):
    """
    Generates datasets for multiple combinations of participants, healthy ratios, and datasets.

    Args:
    biomarker_order (Dict[str, int]): Biomarker names and their orders
    real_theta_phi_file (str): Path to the JSON file containing theta and phi values.
    js (List[int]): List of numbers of participants.
    rs (List[float]): List of healthy ratios.
    num_of_datasets_per_combination (int): Number of datasets to generate per combination.
    output_dir (str): Directory to save the generated datasets.
    seed (Optional[int]): Global seed for reproducibility. If None, a random seed is used.
    stage_distribution (Optional[str]), either 'uniform' or 'dirichlet'
    prefix (Optional[str]): Optional prefix of filename
    suffix (Optional[str]): Optional suffix of filename
    keep_all_cols (Optional[bool]): if False, drop ['k_j', 'S_n', 'affected_or_not']
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if seed is None:
        seed = np.random.SeedSequence().entropy 
    rng = np.random.default_rng(seed)

    for j in js:
        for r in rs:
            for m in range(num_of_datasets_per_combination):
                sub_seed = rng.integers(0, 1_000_000)
                generate_data_from_ebm(
                    n_participants=j,
                    biomarker_order=biomarker_order,
                    real_theta_phi_file=real_theta_phi_file,
                    healthy_ratio=r,
                    output_dir=output_dir,
                    m=m,
                    seed=sub_seed,
                    stage_distribution=stage_distribution,
                    prefix=prefix,
                    suffix=suffix,
                    keep_all_cols = keep_all_cols
                )
    print(f"Data generation complete. Files saved in {output_dir}/")
