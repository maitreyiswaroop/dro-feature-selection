# dro-feature-selection
Code for implementing experiments for Distributionally Robust Feature Selection.

## Core Scripts

*   **Data Handling**: `data.py`, `data_acs.py`, `data_baseline_failures.py`, `data_uci.py`
*   **Main Experiment Logic**: `gd_pops_v8.py` (for synthetic/general), `gd_pops_v10.py` (for ACS)
*   **Experiment Runners**: `run_synth_expt.sh`, `run_uci_expt.sh`, `run_acs_expt.sh`.
    *   These scripts are configured for single runs with specific seeds and learning rates.
*   **Supporting Modules**: `estimators.py`, `baselines.py`, `downstream_models.py`, `lr_schedulers.py`, `visualisers.py`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd dro-feature-selection
    ```
2.  **Set up Python environment:**
    It's recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    # Example with conda
    conda create -n dro_fs_env python=3.8
    conda activate dro_fs_env
    ```
3.  **Install dependencies:**
    (Assuming a requirements.txt file will be added. If not, list key dependencies here.)
    ```bash
    # pip install -r requirements.txt 
    # Add key libraries like: pip install numpy pandas scikit-learn xgboost torch ...
    ```

## Running Experiments

The primary way to run experiments is through the provided shell scripts:

*   **Synthetic Data Experiment (Baseline Failure 5):**
    ```bash
    bash run_synth_expt.sh
    ```
    This script utilizes `gd_pops_v8.py` via `gd_pops_v8_task.sh`.

*   **UCI Dataset Experiment:**
    ```bash
    bash run_uci_expt.sh
    ```
    This script likely uses `gd_pops_v8.py` (or a similar version adapted for UCI data).

*   **ACS Dataset Experiment:**
    ```bash
    bash run_acs_expt.sh
    ```
    This script utilizes `gd_pops_v10.py`.

Each script is pre-configured for a single run (specific seed, learning rate). Modify the respective shell script or the underlying Python scripts/config files for different parameters or multiple runs.

## Configuration

Experiment parameters might be defined within:
*   The Python scripts (`gd_pops_v*.py`)
*   The runner shell scripts (`run_*.sh`)

Refer to the specific scripts for details on how parameters are set.

## Results

*   Results are typically saved in directories specified within the scripts (e.g., `SAVE_PATH` variable).
*   Logs for SLURM jobs (if used) are specified in the SBATCH directives within the shell scripts.
*   `aggregate_results.py` can be used to collect and summarize


## Project Structure

```
dro-feature-selection/
├── README.md                   # This file
├── data_cache/                 # Cached data to speed up processing
├── data_uci/                   # Data specific to UCI datasets
├── acs_analysis_output/        # Output from ACS data analysis
├── uci_analysis_output/        # Output from UCI data analysis
├── modules/                    # Utility modules or helper functions
├── data.py                     # Main data loading and preprocessing script
├── data_acs.py                 # ACS specific data handling
├── data_baseline_failures.py   # Data handling for baseline failure scenarios
├── data_uci.py                 # UCI specific data handling
├── baselines.py                # Implementation of baseline models/methods
├── estimators.py               # Custom estimators used in experiments
├── downstream_models.py        # Models used for downstream tasks after feature selection
├── global_vars.py              # Global variables or constants
├── lr_schedulers.py            # Learning rate scheduler implementations
├── aggregate_results.py        # Script to aggregate results from multiple runs
├── visualisers.py              # Plotting and visualization utilities
├── visualize_training.py       # Scripts to visualize training progress
|
├── gd_pops_v8.py               # Gradient-based population optimization v8 (synthetic/general)
├── gd_pops_v10.py              # Gradient-based population optimization v10 (ACS)
|
├── run_synth_expt.sh           # Script to run synthetic data experiments (e.g., baseline failures)
├── run_uci_expt.sh             # Script to run UCI dataset experiments
├── run_acs_expt.sh             # Script to run ACS dataset experiments
├── gd_pops_v8_task.sh          # Task script called by synthetic experiment runner
|
├── .git/                       # Git repository data
└── .gitignore                  # Specifies intentionally untracked files that Git should ignore
```

