import numpy as np 
import pandas as pd 
from typing import List, Tuple, Dict
from kde_ebm import mixture_model

import warnings

# Suppress the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message="Values in x were outside bounds during a minimize step*")

def extract_info(df:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Obtain X, y, and biomarker labels from the data

    Args:
        df: pd.DataFrame

    Returns:
        - data: biomarker matrix [n_subjects, n_biomarkers]
        - target: binary disease status [n_subjects]
        - biomarker_labels: list of biomarker names
    """
    diseased_dict = dict(zip(df.participant, df.diseased))

    dff = df.pivot(index='participant', columns='biomarker', values='measurement')
    dff = dff.sort_index(axis=1, level=1, sort_remaining=False)
    dff.columns.name = None
    dff.reset_index(inplace=True)
    dff['diseased'] = [int(diseased_dict[x]) for x in dff.participant]
    dff.drop(columns=['participant'], inplace=True)

    biomarker_labels = list(dff.columns)[:-1]
    data_matrix = dff.to_numpy()
    data = data_matrix[:, :-1]
    target = data_matrix[:, -1].astype(int)

    return data, target, biomarker_labels

def get_gmm_theta_phi_estimates(df:pd.DataFrame) -> Dict[str, Dict[str, np.float64]]:
    """Obtain the theta phi estimates using UCL GMM 
    """
    data, target, biomarker_labels = extract_info(df)
    mixture_models = mixture_model.fit_all_gmm_models(data, target)
    estimates = {}
    for i, model in enumerate(mixture_models):
        estimates[biomarker_labels[i]] = {
            'theta_mean': model.ad_comp.mu,
            'theta_std': model.ad_comp.sigma,
            'phi_mean': model.cn_comp.mu,
            'phi_std': model.cn_comp.sigma 
        }
    return estimates
