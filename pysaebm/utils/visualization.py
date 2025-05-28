import os 
import seaborn as sns
import matplotlib.pyplot as plt
from . import data_processing
from typing import List, Dict, Optional

import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_heatmap(
    all_dicts: List[Dict],
    burn_in:int, 
    thining:int, 
    folder_name:str, 
    file_name:str, 
    title:str,
    best_order: Optional[Dict[str, int]] = None,
    ):
    os.makedirs(folder_name, exist_ok=True)
    
    biomarker_stage_probability_df = data_processing.get_biomarker_stage_probability(
        all_dicts, burn_in, thining
    )

    if best_order:
        biomarker_order = dict(sorted(best_order.items(), key=lambda item:item[1]))
        ordered_biomarkers = list(biomarker_order.keys())

        # Rename index to include order in format "ABC (1)"
        renamed_index = [f"{biomarker} ({biomarker_order[biomarker]})" for biomarker in ordered_biomarkers]

        # Reorder DataFrame rows
        biomarker_stage_probability_df = biomarker_stage_probability_df.loc[ordered_biomarkers]
        biomarker_stage_probability_df.index = renamed_index
    
    # Find the longest biomarker name
    max_name_length = max(len(name) for name in biomarker_stage_probability_df.index)
    
    # Dynamically adjust figure width based on the longest name
    fig_width = max(10, max_name_length * 0.3)  # Scale width based on name length
    
    plt.figure(figsize=(fig_width, 8))  # Increase width to accommodate long names
    
    sns.heatmap(
        biomarker_stage_probability_df,
        annot=True, 
        cmap="Blues", 
        linewidths=.5,
        cbar_kws={'label': 'Probability'},
        fmt=".2f",
        vmin=0,
        vmax=1
    )
    
    plt.xlabel('Stage Position')
    plt.ylabel('Biomarker')
    plt.title(title)
    
    # Adjust y-axis ticks to avoid truncation
    plt.yticks(rotation=0, ha='right')  # Ensure biomarker names are horizontal and right-aligned
    
    # Adjust left margin if names are still getting cut off
    plt.subplots_adjust(left=0.3)  # Increase left margin (default is ~0.125)

    plt.tight_layout()
    
    # Save figure with padding to ensure labels are not cut off
    plt.savefig(f"{folder_name}/{file_name}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{folder_name}/{file_name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()

def save_traceplot(
    log_likelihoods: List[float],
    folder_name: str,
    file_name: str,
    title: str,
    skip: int = 40,
    upper_limit: Optional[float] = None, 
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