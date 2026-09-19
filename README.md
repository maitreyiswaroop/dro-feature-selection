# Distributionally Robust Feature Selection

Code for the experiments in **Distributionally Robust Feature Selection** ([paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/5b3f001200198bb4ecaac0a1ea89dd99-Paper-Conference.pdf), NeurIPS 2025).

We study model-agnostic feature selection across multiple populations. The method learns a stochastic feature-degradation mask and minimizes worst-population prediction risk. Under squared loss, this risk can be expressed using the expected conditional variance of the outcome given the degraded features.

## Code map

- `estimators.py`: conditional-mean, kernel, Monte Carlo, and influence-function estimators. The optimized kernel implementation is in `estimate_conditional_keops_flexible_optimized`, and the corresponding objective estimator is in `estimate_T2_kernel_IF_like_flexible`.
- `gd_pops_v8.py`: primary experiment driver for the synthetic and UCI experiments.
- `gd_pops_v10.py`: modular, checkpointed experiment driver used for ACS experiments.
- `modules/`: data loading, checkpointing, feature selection, baselines, and downstream evaluation for the modular pipeline.
- `data.py`, `data_acs.py`, `data_baseline_failures.py`, `data_uci.py`: data generation and preprocessing.
- `baselines.py` and `downstream_models.py`: comparison methods and downstream evaluation.
- `aggregate_results.py` and `visualize_training.py`: result aggregation and plots.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/maitreyiswaroop/dro-feature-selection.git
cd dro-feature-selection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running experiments

The supplied runners reproduce one representative configuration for each dataset family:

```bash
bash run_synth_expt.sh
bash run_uci_expt.sh
bash run_acs_expt.sh
```

The scripts contain SLURM directives and cluster-specific environment setup. Before running them on another machine or cluster, update the partition, resource requests, and Conda activation lines. Output directories and the main experimental parameters are defined near the top of each runner.

For direct invocation and the full list of arguments:

```bash
python gd_pops_v8.py --help
python gd_pops_v10.py --help
```

Generated datasets, logs, and experiment outputs are intentionally excluded from version control.
