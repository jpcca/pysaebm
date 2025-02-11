import os 
import seaborn as sns
import matplotlib.pyplot as plt
from . import data_processing
from typing import List, Dict

import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_heatmap(all_dicts, burn_in, thining, folder_name, file_name, title):
    os.makedirs(folder_name, exist_ok=True)
    
    biomarker_stage_probability_df = data_processing.get_biomarker_stage_probability(
        all_dicts, burn_in, thining
    )
    
    # Find the longest biomarker name
    longest_biomarker = max(biomarker_stage_probability_df.index, key=len)
    max_name_length = len(longest_biomarker)
    
    # Dynamically adjust figure width based on the longest name
    fig_width = max(10, max_name_length * 0.3)  # Scale width based on name length
    
    plt.figure(figsize=(fig_width, 8))  # Increase width to accommodate long names
    
    sns.heatmap(biomarker_stage_probability_df,
                annot=True, cmap="Greys", linewidths=.5,
                cbar_kws={'label': 'Probability'},
                fmt=".1f"
    )
    
    plt.xlabel('Stage')
    plt.ylabel('Biomarker')
    plt.title(title)
    
    # Adjust y-axis ticks to avoid truncation
    plt.yticks(rotation=0, ha='right')  # Ensure biomarker names are horizontal and right-aligned
    
    # Adjust left margin if names are still getting cut off
    plt.subplots_adjust(left=0.3)  # Increase left margin (default is ~0.125)

    plt.tight_layout()
    
    # Save figure with padding to ensure labels are not cut off
    plt.savefig(f"{folder_name}/{file_name}.png", bbox_inches="tight", dpi=300)
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