import os 
import seaborn as sns
import matplotlib.pyplot as plt
from . import data_processing
from typing import List, Dict

def save_heatmap(all_dicts, burn_in, thining, folder_name, file_name, title):
    os.makedirs(folder_name, exist_ok=True)
    biomarker_stage_probability_df = data_processing.get_biomarker_stage_probability(
        all_dicts, burn_in, thining)
    sns.heatmap(biomarker_stage_probability_df,
                annot=True, cmap="Greys", linewidths=.5,
                cbar_kws={'label': 'Probability'},
                fmt=".1f"
    )
    plt.xlabel('Stage')
    plt.ylabel('Biomarker')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/{file_name}.png")
    plt.close()


def save_traceplot(
    log_likelihoods: List[float],
    folder_name: str,
    file_name: str):
    os.makedirs(folder_name, exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(log_likelihoods, label="Log Likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.title("Trace Plot of Log Likelihood")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder_name}/{file_name}.png")
    plt.close()