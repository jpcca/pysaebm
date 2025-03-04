import re 
import os 
import logging 

def extract_fname(data_file:str) -> str:
    """
    Extract the base filename (without extension) from the data file path.
    Replace invalid characters (e.g., `|`) with underscores.
    """
    base_name = data_file.split('/')[-1]
    fname = base_name.split(".csv")[0]
    fname = re.sub(r'[\\|/:"*?<>]+', '_', fname)
    return fname 

def cleanup_old_files(output_dir: str, fname: str):
    """
    Remove old files (heatmap, traceplot, results, log) for the given fname.
    Logs a warning if files do not exist.
    """
    files_to_remove = [
        f"{output_dir}/heatmaps/{fname}_heatmap.png",
        f"{output_dir}/traceplots/{fname}_traceplot.png",
        f"{output_dir}/results/{fname}_results.json",
        f"{output_dir}/logs/{fname}.log"
    ]

    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Removed old file: {file_path}")
            except Exception as e:
                logging.error(f"Error removing old file: {file_path}: {e}")
        else:
            logging.warning(f"File does not exist, skipping removal: {file_path}")