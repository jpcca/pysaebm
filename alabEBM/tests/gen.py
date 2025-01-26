from alabEBM import generate, get_params_path
import os

# Get path to default parameters
params_file = get_params_path()

# Generate data using default parameters
S_ordering = [
    'HIP-FCI', 'PCC-FCI', 'AB', 'P-Tau', 'MMSE', 'ADAS',
    'HIP-GMI', 'AVLT-Sum', 'FUS-GMI', 'FUS-FCI'
]

generate(
    S_ordering=S_ordering,
    real_theta_phi_file=params_file,  # Use default parameters
    js = [50, 100],
    rs = [0.1, 0.5],
    num_of_datasets_per_combination=2,
    output_dir='my_data'
)