# `pysaebm`

`pysaebm` is a Python package for generating and analyzing biomarker data using Stage-Aware Event-Based Modeling (SA-EBM). It supports various data generation experiments and EBM algorithms to estimate biomarker orderings and disease stages. This package is designed for researchers and data scientists working with biomarker progression analysis.

For detailed methodology, refer to [our paper](https://saebm.hongtaoh.com/).


## Installation

Install `pysaebm` using pip:

```bash
pip install pysaebm
```

Ensure you have Python 3.8+ and the required dependencies installed. For a full list of dependencies, see `requirements.txt`.



## Data generation




Examples of how to generate data are at [./pysaebm/test/gen.py](./pysaebm/test/gen.py).


Because in each generation, the ordering is randomized, you will see a `true_order_and_stages.json` that tells you the corresponding true stages and true order for each output csv file.


The source codes for data generation can be seen in [./pysaebm/utils/generate_data.py](./pysaebm/utils/generate_data.py).


This is the full `generate` parameters:


```py
def generate(
   experiment_name: str = "sn_kjOrdinalDM_xnjNormal",
   params_file: str = 'params.json',
   js: List[int] = [50, 200, 500, 1000],
   rs: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
   num_of_datasets_per_combination: int = 50,
   output_dir: str = 'data',
   seed: Optional[int] = None,
   dirichlet_alpha: Optional[Dict[str, List[float]]] = {
       'uniform': [100],
       'multinomial': [0.4013728324975898,
                       1.0910444770153345,
                       2.30974117596663,
                       3.8081194066281103,
                       4.889722107892335,
                       4.889722107892335,
                       3.8081194066281103,
                       2.30974117596663,
                       1.0910444770153345,
                       0.4013728324975898]
   },
   beta_params: Dict[str, Dict[str, float]] = {
       'near_normal': {'alpha': 2.0, 'beta': 2.0},
       'uniform': {'alpha': 1, 'beta': 1},
       'regular': {'alpha': 5, 'beta': 2}
   },
   prefix: Optional[str] = None,
   suffix: Optional[str] = None,
   keep_all_cols: bool = False,
   fixed_biomarker_order: bool = False,
   noise_std_parameter: float = 0.05,
) -> Dict[str, Dict[str, int]]:
```


Explanations:
- `experiment_name` should be one of the these:


```py
experiment_names = [
  "sn_kjOrdinalDM_xnjNormal",     # Experiment 1: Ordinal kj with Dirichlet-Multinomial, Normal Xnj
  "sn_kjOrdinalDM_xnjNonNormal",  # Experiment 2: Ordinal kj with Dirichlet-Multinomial, Non-Normal Xnj
  "sn_kjOrdinalUniform_xnjNormal", # Experiment 3: Ordinal kj with Uniform distribution, Normal Xnj
  "sn_kjOrdinalUniform_xnjNonNormal", # Experiment 4: Ordinal kj with Uniform distribution, Non-Normal Xnj
  "sn_kjContinuousUniform",       # Experiment 5: Continuous kj with Uniform distribution
  "sn_kjContinuousBeta",          # Experiment 6: Continuous kj with Beta distribution
  "xiNearNormal_kjContinuousUniform", # Experiment 7: Near-normal Xi with Continuous Uniform kj
  "xiNearNormal_kjContinuousBeta", # Experiment 8: Near-normal Xi with Continuous Beta kj
  "xiNearNormalWithNoise_kjContinuousBeta", # Experiment 9: Same as Exp 8 but with noises to xi
]
```


You can find the explanation to these terms from [our paper](https://saebm.hongtaoh.com/).


- `params_file`: The path to the parameters in json. Example is [./pysaebm/data/params.json](./pysaebm/data/params.json). You should specify each biomarker's `theta_mean`, `theta_std`, `phi_mean`, and `phi_std`.
- `js`: An array of integers indicating the number of participants you want.
- `rs`: An array of floats indicating the number of healthy ratios.
- `num_of_datasets_per_combination`: The number of repetitions for each j-r combination.
- `output_dir`: The directory where you want to save the generated data.
- `seed`: An integer serving as the seed for the randomness.
- `dirichlet_alpha`: This should be a dictionary where keys are `uniform` and `multinomial`. They correspond to `kjOrdinalUniform` and `kjOrdinalDM`.
- `beta_params`: A dictionary where keys are `near_normal`, `uniform`, and `regular`, corresponding to `xiNearNormal`, `kjContinuousUniform` and `kjContinuousBeta`.
- `prefix`: Optional prefix for the output csv file names.
- `suffix`: Optional suffix for the output csv file names.
- `keep_all_cols`: Whether to include additional metadata columns (k_j, event_time, affected)
- fixed_biomarker_order: If True, will use the order as in the `params_file`. If False, will randomize the ordering.
- noise_std_parameter: the parameter in N(0, N \cdot noise_std_parameter) in experiment 9


Note that you need to make sure the `dirichlet_alpha['multinomial']` has the same length as your params dict (as in your `params_file`).




## Run EBM Algorithms




Examples of how to run algorithms and get results is at [./pysaebm/test/test.py](./pysaebm/test/test.py).




This explains the parameters well enough:


```py
def run_ebm(
    algorithm:str,
    data_file: str,
    output_dir: str,
    output_folder: Optional[str] = None,
    n_iter: int = 2000,
    n_shuffle: int = 2,
    burn_in: int = 500,
    thinning: int = 1,
    true_order_dict: Optional[Dict[str, int]] = None,
    true_stages: Optional[List[int]] = None,
    plot_title_detail: Optional[str] = "",
    fname_prefix: Optional[str] = "",
    skip_heatmap: Optional[bool] = False,
    skip_traceplot: Optional[bool] = False,
    # Strength of the prior belief in prior estimate of the mean (μ), set to 1 as default
    prior_n: float = 1.0,
    # Prior degrees of freedom, influencing the certainty of prior estimate of the variance (σ²), set to 1 as default
    prior_v: float = 1.0,
    seed: int = 123,
    save_results:bool=True,
    save_theta_phi:bool=False,
    save_stage_post:bool=False,
    save_details:bool=False,
) -> Dict[str, Union[str, int, float, Dict, List]]:
    """
    Run the metropolis hastings algorithm and save results 

    Args:
        algorithm (str): Choose from 'hard_kmeans', 'mle', 'em', 'kde', and 'conjugate_priors' (default).
        data_file (str): Path to the input CSV file with biomarker data.
        output_dir (str): Path to the directory to store all the results.
        output_folder (str): Optional. If not provided, all results will be saved to output_dir/algorithm. 
            If provided, results will be saved to output_dir/output_folder
        n_iter (int): Number of iterations for the Metropolis-Hastings algorithm.
        n_shuffle (int): Number of shuffles per iteration.
        burn_in (int): Burn-in period for the MCMC chain.
        thinning (int): Thinning interval for the MCMC chain.
        true_order_dict (Optional[Dict[str, int]]): biomarker name: the correct order of it (if known)
        true_stages (Optional[List[int]]): true stages for all participants (if known)
        plot_title_detail (Optional[str]): optional string to add to plot title, as suffix.
        fname_prefix (Optional[str]): the prefix of heatmap, traceplot, results.json, and logs file, e.g., 5_50_0_heatmap_conjugate_priors.png
            In the example, there are no prefix strings. 
        skip_heatmap (Optional[bool]): whether to save heatmaps. True you want to skip saving heatmaps and save space.
        skip_traceplot (Optional[bool]): whether to save traceplots. True if you want to skip saving traceplots and save space.
        prior_n (strength of belief in prior of mean): default to be 1.0
        prior_v (prior degree of freedom) are the weakly informative priors, default to be 1.0
        seed (int): for reproducibility
        save_results: whether to save the json result.
        save_theta_phi: if save_results, whether to include the theta_phi result.
        save_stage_post: if save_results, whether to include the stage post result. 
        save_details: if save_results, save the simple version instead of the complete version. 

    Returns:
        Dict[str, Union[str, int, float, Dict, List]]: Results including everything, e.g., Kendall's tau and p-value.

        Whether to save results or not, the results will be returned. If save_details, the complete veresion will be returned.
        Otherwise, the simple version will be returned. 
    """
```


Some extra explanations:




- `n_iter`: In general, above 2k is recommended. 10k should be sufficient if you have <= 10 biomarkers.
- `burn_in` and `thinning`: The idea behind the two parameters is that we will only use some of the results from all iterations in `n_iter`. We will do this: if `(i > burn_in) & (i % thinning == 0)`, then we will use the result from that iteration `i`. Usually, we set `thinning` to be 1.


After running the `run_ebm`, you'll see a folder named as your `output_dir`. Each algorithm will have its subfolders.


The results are organized this way:

- `records` folder contains the loggings.
- `heatmaps` folder contains all the heatmaps. An example is


![An example of heatmap](./heatmap_example.png)


  Biomarkers in the y-axis are ranked according to the ordering that has the highest likelihood among all iterations. Each cell indicates the probability that a certain biomarker falls in a certain stage. **Note, however, these probabilities are calculated based on all the iterations that satisfy  `(i > burn_in) & (i % thinning == 0)`**.


  In the heatmap, the sum of each col and each row is 1.

- `traceplots` folder contains all the traceplots (starting from iteration 40, not iteration 0). Those plots will be useful to diagnose whether EBM algorithms are working correctly. It's totally okay for the plots to show fluctuation (because biomarker distribution and stage distributions are re-calculated each iterations). You should not, however, see a clear downward trend.


![An example of traceplot](./traceplot_example.png)

- `results` folder contains all important results in `json`. Each file contains


```py
results = {
            "algorithm": algorithm, # name of the algoritm
            "runtime": end_time - start_time, # run time in seconds
            "N_MCMC": n_iter, # number of MCMC iterations
            "n_shuffle": n_shuffle, # number of shuffled item in each Metropolis Hastings algorithm
            "burn_in": burn_in, # burn in
            "thinning": thinning, # thinning
            'healthy_ratio': healthy_ratio, # percentage of healthy participants
            "max_log_likelihood": float(max(log_likelihoods)), # # max of log_likelihoods in (N_MCMC - burn_in) & thinning results
            "kendalls_tau": tau, # kendalls tau of the ordering with max log likelihood with the true ordering
            "p_value": p_value,
            "mean_absolute_error": mae, # mae of the staging result
            'current_pi': current_pi.tolist(), # stage prob of stage 1 to N where N is the number of biomarkers
            'updated_pi': updated_pi.tolist(), # updated pi is the pi for all stages, including 0
            'true_order': true_order_result, # ground true of ordering, if available
            "order_with_highest_ll": {k: int(v) for k, v in zip(biomarker_names, order_with_highest_ll)},
            "true_stages": true_stages, # ground true of stages, if available
            'ml_stages': ml_stages, # most likely stages, according to updated_pi
            "stage_likelihood_posterior": final_stage_post_dict, # stage distribution posterior 
            "final_theta_phi_params": final_theta_phi_dict, # the final theta phi params for each biomarker
        }
```


- `algorithm` is the algorithm you are running with.
- `runtime` in seconds.
- `N_MCMC` is the number of MCMC iterations.
- `n_shuffle` is how many biomarkers to shuffle places in Metropolis Hastings. Default is 2.
- `burn_in` and `thinning` are explained above.
- `healthy_ratio` is the percentage of non-diseased participants in the dataset.
- `max_log_likelihood` is the max data log likelihood. The script [./pysaebm/algorithms/algorithm.py](./pysaebm/algorithms/algorithm.py) generates `all_accepted_orders`, `log_likelihoods`, `current_theta_phi`, `current_stage_post`, and `current_pi`. `max_log_likelihoods` is the max of `log_likelihoods`.
- `kendalls_tau` and `p_value` are the result of comparing the `ml_order` (most likely order) with the true order (if provided).
- `current_pi` is the probability distribution of all disease stages.
- `current_pi` is the probability distribution of all stages, including 0.
- `true_order` is the dictionary explaining the stage each biomarker is at.
- `ml_order` is the overall most likely order considering results from all the iterations that satisfy  `(i > burn_in) & (i % thinning == 0)`. It is calculated by the function of `obtain_most_likely_order_dic` in [./pysaebm/utils/data_processing.py](./pysaebm/utils/data_processing.py)
- `order_with_highest_ll` is the ordering that corresponds to the `max_log_likelihoods`.
- `true_stages` is an array of each participant's disease stage.
- `ml_stages` is the array of most likely disease stages.
- `stage_likelihood_posterior` is a dictionary detailing each participant's probability in each of the possible stages.
- - `stage_likelihood_posterior_diseased` is a dictionary detailing each diseased participant's probability in each of the possible disease stages (no 0).
- `final_theta_phi_params` is a dictionary detailing the parameters for each biomarker. If you are using `kde`, you'll see data points and `theta_weights` and `phi_weights`. If you use other algorithms, you will see the `theta_mean`, `theta_std`, `phi_mean` and `phi_std`.




## Use your own data


You are more than welcome to use your own data. After all, the very purpose of `pysaebm` is: to allow you to analyze your own data. However, you do have to make sure that the input data have at least four columns:




- participant: int
- biomarker: str
- measurement: float
- diseased: bool


The `participant` column should be integers from 0 to `J-1` where `J` is the number of participants.


Samples are available at [./pysaebm/data/samples/](./pysaebm/data/samples/).


The data should be in a [tidy format](https://vita.had.co.nz/papers/tidy-data.pdf), i.e.,




- Each variable is a column.
- Each observation is a row.
- Each type of observational unit is a table.


## Change Log




- 2025-02-26 (V 0.3.4).
  - Modified the `shuffle_order` function to ensure full derangement, making convergence faster.
- 2025-03-06 (V 0.4.0)
  - use `pyproject.toml` instead
  - update `conjuage_priors_algo.py`, now without using the auxiliary variable of `participant_stages`. Keep the uncertainties just like in `soft_kmeans_algo.py`.
- 2025-03-07 (V 0.4.2)
  - Compute `new_ln_likelihood_new_theta_phi` based on `new_theta_phi_estimates`, which is based on `stage_likelihoods_posteriors` that is based on the newly proposed order and previous `theta_phi_estimates`.
  - Update `theta_phi_estimates` with `new_theta_phi_estimates` only if a new order is accepted.
  - The fallback theta_phi_estimates is the previous parameters rather than theta_phi_default
  - `all_accepted_orders.append(current_order_dict.copy())` to make sure the results are not mutated.
  - Previously I calculated the `new_ln_likelihood` and `stage_likelihoods_posteriors` based on the newly proposed order and previous `theta_phi_estimates`, and directly updated theta_phi_estimates whether we accept the new order or not.
  - Previously, I excluded `copy()` in `all_accepted_orders.append(current_order_dict.copy())`, which is inaccurate.
- 2025-03-17 (V 0.4.3)
  - Added `skip` and `title_detail` parameters in the `save_traceplot` function.
- 2025-03-18 (V 0.4.4)
  - Add an optional horizontal bar indicating the upper limit in the trace plot.
- 2025-03-18 (V 0.4.7)
  - Allowed keeping all cols (`keep_all_cols`) in data generation.
- 2025-03-18 (V 0.4.9)
  - copy `data_we_have` and use `data_we_have.loc[:, 'S_n']` in soft k means algo when preprocessing participant and biomarker data.
- 2025-03-20 (V 0.5.1)
  - In hard kmeans, updated `delta = ln_likelihood - current_ln_likelihood`, and in soft kmeans and conjugate priors, made sure I am using `delta = new_ln_likelihood_new_theta_phi - current_ln_likelihood`.
  - In each iteration, use `theta_phi_estimates = theta_phi_default.copy()` first. This means, `stage_likelihoods_posteriors` is based on the default theta_phi, not the previous iteration.
- 2025-03-21 (V 0.6.0)
  - Integrated all three algorithms to just one file `algorithms/algorithm.py`.
  - Changed the algorithm name of `soft_kmeans` to `mle` (maximum likelihood estimation)
  - Moved all helper functions from the algorithm script to `utils/data_processing.py`.
- 2025-03-22 (V 0.7.6)
  - Current state should include both the current accepted order and its associated theta/phi. When updating theta/phi at the start of each iteration, use the current state's theta/phi (1) in the calculation of stage likelihoods and (2) as the fallback if either of the biomarker's clusters is empty or has only one measurement; (3) as the prior mean and variance.
  - Set `conjugate_priors` as the default algorithm.
  - (Tried using cluster's mean and var as the prior but the results are not as good as using current state's theta/phi as the prior).
- 2025-03-24 (V 0.7.8)
  - In heatmap, reorder the biomarkers according to the most likely order.
  - In `results.json` reorder the biomarker according to their order rather than alphabetically ranked.
  - Modified `obtain_most_likely_order_dic` so that we assign stages for biomarkers that have the highest probabilities first.
  - In `results.json`, output the order associated with the highest total log likelihood. Also, calculate the kendall's tau and p values of it and the original order (if provided).
- 2025-03-25 (V 0.8.1)
  - In heatmap, reorder according to the order with highest log likelihood. Also, add the number just like (1).
  - Able to add title detail to heatmaps and traceplots.
  - Able to add `fname_prefix` in `run_ebm()`.
- 2025-03-29 (V 0.8.9)
  - Added `em` algorithm.
  - Added Dirichlet-Multinomial Model to describe uncertainty of stage distribution (a multinomial distribution of all disease stages; because we cannot always assume all disease stages are equally likely).
  - `prior_v` default set to be 1.
  - Default to use dirichlet distribution instead of uniform distribution
  - Change data filename from 50|100_1 to 50_100_1.
  - Modified the `mle` algorithm to make sure the output does not contain `np.nan` (by using the fallback).




- 2025-03-30 (V 0.9.2)
  - Completed change to `generate_data.py`. Now incorporates the modified data generation model based on DEBM2019.
  - Rank the original order by the value (ascending), if the original order exists.
  - Able to skip saving traceplots and/or heatmaps.




- 2025-03-31 (V 0.9.4)
  - Able to store final theta phi estimates and the final stage likelihood posterior to results.json




- 2025-04-02 (V 0.9.5)
  - Added `kde` algorithm.
  - Initial kmeans used seeded Kmeans + conjugate priors.




- 2025-04-03 (V 0.9.7)
  - Improved kde.
  - Added dirichlet and beta parameters randomization.




- 2025-04-05 (V 0.9.9)
  - Updated `generate_data.py` to align with the experimental design.




- 2025-04-06 (V 0.9.9.3)
  - Make `kmeans.py` more robust. Now try 100 times to randomize the assignment for the diseased group if the initial kmeans failed.
  - Add `algorithm` and `ml_order` in `results.json`
  - After generating data, create `true_order_and_stages_dict` to store all filenames' true biomarker order and all participants' stages. For continuous kjs, use the `bisect_right` algorithm to get the ranking order.




- 2025-04-07 (V 0.9.9.9)
  - Modified `generate_data.py` to allow Experiment 9.
  - In `generate_data.py`, added the function of randomly flipping the direction of progression in the sigmoid model. Also make sure this random direction is consistent across participants.
  - In the spirit of "Do not repeat yourself", delete "R" and "rho" in params.json. Instead, compute it each time when I generate data. The time difference is minimal.
  - Added comparisons between true stages and most likely stages
  - Reorganized the results.json
  - Allow `output_dir` in `run_ebm`.


- 2025-04-08 (V 1.00)
  - Modify the FastKDE implementation.
  - Reorganized the `results.json`.
  - Added `stage_post` to hard kmeans as well. Now every algorithm can output ml_stages.
  - Make sure diseased stages are between `(, max_stage]`.


- 2025-04-13 (V 1.1)
  - Leveraged weights when calculating bandwidth in `FastKDE`.


- 2025-04-14 (V 1.11)
  - Updated the `fastkde.py`.


- 2025-04-15 (V 1.131)
  - Renamed the package to `pypysaebm`.
  - Reconstructed the `README` documentation.


- 2025-04-16 (V 2.0.1)
  - Did the `fast_kde.py` on my own. Streamlined the script quite a bit. Now it's very readable and easy to follow & understand.
  - Used scott for bandwidth selection


- 2025-04-16 (V 2.0.2)
  - Checked whether the length of parameters is the same as the dirichlet_alpha multinomial.
  - Added an option to use fixed biomarker order in data generation.


- 2025-04-19 (V 2.0.3)
  - Anonymize the readme on Pypi


- 2025-04-20
  - Modify the format of the result of run.py to be `Dict[str, Union[str, int, float, Dict, List]]`.


- 2025-04-21
  - Use `np.random.choice(len(final_stage_post[pid]), p=final_stage_post[pid]) + 1` to obtain `ml_stages` instead of `argmax`.


- 2025-04-27 (V 2.0.2)
  - Make the noise std as a parameter in experiment 9.
  - Added `current_pi` to the output results json.


- 2025-04-30 (V 2.0.3)
  - Added the `gmm` algorithm.


- 2025-05-01 (V 2.0.5)
  - True uniform kjs now in data generation.


- 2025-05-02 (V 2.0.6)
  - Distinguished between 'gmm' and 'gmm_dm_prior'.


- 2025-05-03 (V 2.0.9)
  - Distinguished between 'conjugate_priors' and "conjugate_priors_gm_prior".


- 2025-05-07 (V 2.1.0)
  - Kept only the original five algorithms.
  - In `run.py`, make sure to check whether true_order exists or not before saving to json.


- 2025-05-11 (V 2.1.1)
  - Updated data generation, now using approximate uniform in kj generation.


- 2025-05-12 (V 2.1.3)
  - Added stage prediction for healthy participants as well, which means for each and every participant, when referencing stage, all stages (including 0) will be tested.
  - Updated heatmap visualization design; using blues color map now.
  - Save both png and pdf for heatmap.


- 2025-05-13 (V 2.1.4)
  - Added PDF for traceplot save.
  - Improved heatmap title.


- 2025-05-18 (V 2.1.6)
  - ml_stages, both for all and for diseased only, are based on `participant_data` that is using `order_with_highest_ll`.
  - Flipped the sign of mean of phi and theta for adas, p-tau, hip-fci, and fus-fci in `data/params.json`.


- 2025-05-28 (V 2.2.0)
  - Added algorithm name in the title of traceplots.
  - Added runtime in `results.json`.


- 2025-05-29 (V 2.2.2)
  - Added seed in `run.py` to enable reproducibility of the ordering results.
  - Added `rng` in random shuffling.


- 2025-06-21 (V 2.2.3)
  - Make sure that for kde, in `calculate_bandwidth`, set a lower bound for sigma.


- 2025-06-22 (V 2.2.5)
  - Added mixed pathology functions in `generate_data.py`, `data_processing.py`, `algorithm.py` and `run.py`.
  - Added a parameter of `output_folder` in `run.py`.


- 2025-06-24 (V 2.2.6)
  - Added `max_log_likelihood` in results json in `run.py`.


- 2025-06-29 (V 2.2.11)
  - Enabled Generalized Mallows as an alternative to Pairwise Preferences in energy and combined ordering generation.
  - Fixed the bug of `params` in generating data.


- 2025-06-30 (V 2.2.12)
  - Fixed typo: `unbiased`.


---


All above changelogs were [alabebm](https://pypi.org/project/alabebm/#history). However, I changed the name to [pysaebm](https://pypi.org/project/pysaebm/) after the paper got accepted. This is because of the required anonymity.


Below will be the changelogs for `pysaebm`.


- 2025-07-09 (V 1.0.1)
  - Made `mp_method` optional in `generate_data.py`.
  - Made `mp_method` and `order_array` in `algorithm.py` and `run.py` as optional.


- 2025-08-01 (V 1.0.2)
  - Deleted `mixed_pathology` stuff.

- 2025-08-03 (V 1.1.0)
  - changed from `np.log(current_pi[k_j-1])` to `np.log(current_pi[k_j])` in unbaised stage post. 

- 2025-08-06 (V 1.2.1)
  - Used vectorized version for all algos except for kde. Used the pd.dataframe and dicts for kde.
  - Used tau distance instead of tau corr in saved results.
  - Made sure `fastmath=False` in each `njit`.
  
- 2025-08-06 (V 1.2.4)
  - Solved the logic bug of `save_details` and `save_results`.

- 2025-08-18 (V 1.2.6)
  - Used a np seed in the function of `obtain_affected_and_non_clusters`.

- 2025-08-19 (V 1.2.9)
  - Corrected an error: in data generation, for experiment 9, the noise_std should be max_length * noise_std_parameter rather than its square root. I don't need to redo the experiments because after using square root, the noise_std in fact become larger, not smaller. For example, in our example where N = 10, the noise_std should be N*0.05 = 0.5, but after square root, it becomes 0.7. 
  - Changed 

```py
def compute_unbiased_stage_likelihoods(
    n_participants:int,
    data_matrix:np.ndarray,
    new_order:np.ndarray,
    theta_phi: np.ndarray,
    updated_pi: np.ndarray,
    n_stages=int,
) -> np.ndarray:
```

to 

```py
def compute_unbiased_stage_likelihoods(
    n_participants:int,
    data_matrix:np.ndarray,
    new_order:np.ndarray,
    theta_phi: np.ndarray,
    updated_pi: np.ndarray,
    n_stages: int,
) -> np.ndarray:
```

- 2025-08-21 (V 1.3.1)
    - Update the staging task algorithm in `run.py` for kde and other algos.

- 2025-09-02 (V 1.3.7)
    - Updated the conjugate priors. 
    - updated the `mh.py` and `kde_mh.py` about what to return. 
    - Now using the best_theta_phi and best_stage_prior to get the updated_stage_post (when we are blind about the cn/ad labels to decide the disease stages).
    - Now when getting the stage assignments, I am totally blind, even to `healthy_ratio`. 
    - Added `iterations >= burn_in` in `mh.py` and `kde_mh.py`. 
- 2025-09-15 (V 7.2)
    - No changes. But accidentally increased the version to 7.1. That's okay. 
    - Added `max_iter_staging` in `run.py`. 
- 2025-09-20 (V 7.5)
    - Made sure `alpha_prior` is a numpy array, not a python list. 
    - Removed `iteration >= burn_in` when updating best_*. 
- 2025-10-08 (V 7.6)
    - Try soft assignment for conjugate priors and compare results. 