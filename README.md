# Distributionally Robust Feature Selection

Code for the experiments in **Distributionally Robust Feature Selection** ([paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/5b3f001200198bb4ecaac0a1ea89dd99-Paper-Conference.pdf), NeurIPS 2025).

We study model-agnostic feature selection in the group-DRO setting. The method learns a stochastic feature-degradation mask and minimizes worst-population prediction risk. Under squared loss, this risk can be expressed using the expected conditional variance of the outcome given the degraded features.

## Code map

- `dro_feature_selection/kernel_estimators.py`: differentiable kernel, Monte Carlo, and influence-function-based objective estimators. The batched kernel implementation is `estimate_conditional_kernel_batched`, and its corresponding objective estimator is `estimate_T2_kernel_IF_like_flexible`.
- `dro_feature_selection/estimators.py`: conditional-mean and gradient estimators used by the paper experiments.
- `run_experiment.py`: unified, checkpointed experiment driver for synthetic, UCI, and ACS experiments.
- `dro_feature_selection/`: importable package containing data loading, checkpointing, feature selection, estimators, baselines, evaluation, and visualization.
- `dro_feature_selection/data/`: synthetic, UCI, ACS, and baseline-failure dataset implementations.
- `scripts/slurm/`: Bash/SLURM launchers for the paper experiment families.
- `scripts/analysis/`: result-processing utilities.

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

The UCI and ACS runners use the saved settings from the paper experiments. The
synthetic runner provides a representative baseline-failure experiment:

```bash
bash scripts/slurm/run_synthetic.sh
bash scripts/slurm/run_uci.sh
bash scripts/slurm/run_acs.sh
```

The scripts contain SLURM directives and cluster-specific environment setup. Before running them on another machine or cluster, update the partition, resource requests, and Conda activation lines. Output directories and the main experimental parameters are defined near the top of each runner.

For direct invocation and the full list of arguments:

```bash
python run_experiment.py --help
```

Generated datasets, logs, and experiment outputs are intentionally excluded from version control.
