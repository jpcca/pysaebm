from alabebm import generate, get_params_path
import os
import json 

# Get path to default parameters
params_file = get_params_path()

frameworks = ['discrete', 'sigmoid']
stage_distributions = ["continuous_uniform", "continuous_beta", "discrete_uniform",  "discrete_dirichlet"]

for stage_distribution in stage_distributions:
    generate(
            framework='sigmoid',
            params_file=params_file,
            js = [200],
            rs = [0.25],
            num_of_datasets_per_combination=10,
            output_dir='my_data',
            seed=42,
            stage_distribution=stage_distribution
        )

# for framework in frameworks:
#     if framework == 'discrete':
#         for stage_distribution in stage_distributions[2:]:
#             generate(
#                 framework=framework,
#                 params_file=params_file,
#                 js = [50, 100],
#                 rs = [0.1, 0.5],
#                 num_of_datasets_per_combination=2,
#                 output_dir='my_data',
#                 seed=42,
#                 stage_distribution=stage_distribution
#             )
#     else:
#         for stage_distribution in stage_distributions:
#             generate(
#                 framework=framework,
#                 params_file=params_file,
#                 js = [50, 100],
#                 rs = [0.1, 0.5],
#                 num_of_datasets_per_combination=2,
#                 output_dir='my_data',
#                 seed=42,
#                 stage_distribution=stage_distribution
#             )