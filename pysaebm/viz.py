import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from typing import List

def get_biomarker_stage_probability(
        accepted_orders: List[np.ndarray],
        burn_in:int,
        thinning:int
) -> np.ndarray:
    """
    Compute stage probabilities using NumPy only.
    Input: list of NumPy arrays, each of shape (n_biomarkers,)
    Output: 2D NumPy array of shape (n_biomarkers, n_stages)
            where [i, j] is P(biomarker_i in stage j+1)
    """
    arr = np.array(accepted_orders) # shape: (n_samples, n_biomarkers)
    filtered = arr[(np.arange(len(arr)) > burn_in) & (np.arange(len(arr)) % thinning ==0)]
    n_samples, n_biomarkers = filtered.shape 
    n_stages = n_biomarkers # stages: 1, ..., n_biomarkers

    probs = np.zeros((n_biomarkers, n_stages))

    for bm_idx in range(n_biomarkers):
        # all results for this bm 
        bm_history = filtered[:, bm_idx]
        for stage in range(1, n_stages + 1):
            probs[bm_idx, stage - 1] = np.sum(bm_history == stage) / n_samples
    
    return probs # shape: (n_biomarkers, n_stages)

def save_heatmap(
        accepted_orders:List[np.ndarray],
        burn_in:int,
        thinning:int,
        folder_name:str, 
        file_name:str,
        title:str, 
        biomarker_names:np.ndarray,
        best_order: np.ndarray
):
    os.makedirs(folder_name, exist_ok = True)
    probs = get_biomarker_stage_probability(accepted_orders, burn_in, thinning)
    sorted_indices = np.argsort(best_order)
    probs = probs[sorted_indices]
    biomarker_names = [f"{biomarker_names[i]} ({best_order[i]})" for i in sorted_indices]

    # Adjust width based on max biomarker name
    max_name_len = max(len(name) for name in biomarker_names)
    fig_width = max(10, max_name_len * 0.3)

    plt.figure(figsize=(fig_width, 8))
    sns.heatmap(
        probs,
        annot=True,
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Probability"},
        fmt=".2f",
        vmin=0,
        vmax=1,
        xticklabels=np.arange(1, probs.shape[1]+1),
        yticklabels=biomarker_names
    )
    plt.xlabel("Stage Position")
    plt.ylabel("Biomarker")
    plt.title(title)
    plt.yticks(rotation=0, ha="right")
    plt.tight_layout()
    plt.savefig(f"{folder_name}/{file_name}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{folder_name}/{file_name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

def save_traceplot(
    log_likelihoods: List[float],
    folder_name: str,
    file_name: str,
    title: str,
    skip: int = 40,
    upper_limit: float= None, 
):
    os.makedirs(folder_name, exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(range(skip, len(log_likelihoods)), log_likelihoods[skip:], label="Log Likelihood")
    # Add horizontal line for upper limit if provided
    if upper_limit is not None:
        plt.axhline(y=upper_limit, color='r', linestyle='-', label="Upper Limit")
        
        # Add text annotation for the upper limit
        # Position the text near the right end of the plot with a slight vertical offset
        text_x = len(log_likelihoods) - skip - 5  # 5 points from the right edge
        text_y = upper_limit + 0.02 * (max(log_likelihoods[skip:]) - min(log_likelihoods[skip:]))  # Small vertical offset
        plt.text(text_x, text_y, "Upper Limit", color='r', fontweight='bold')
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder_name}/{file_name}.png", dpi=300)
    plt.savefig(f"{folder_name}/{file_name}.pdf", dpi=300)
    plt.close()