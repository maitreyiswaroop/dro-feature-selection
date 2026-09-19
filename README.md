# Distributionally Robust Feature Selection

Code for the experiments in **Distributionally Robust Feature Selection** ([paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/5b3f001200198bb4ecaac0a1ea89dd99-Paper-Conference.pdf), NeurIPS 2025).

We study model-agnostic feature selection across multiple populations. The method learns a stochastic feature-degradation mask and minimizes worst-population prediction risk. Under squared loss, this risk can be expressed using the expected conditional variance of the outcome given the degraded features.

## Code map

- `dro_feature_selection/kernel_estimators.py`: differentiable kernel, Monte Carlo, and influence-function-like objective estimators. The optimized kernel implementation is `estimate_conditional_keops_flexible_optimized`, and its corresponding objective estimator is `estimate_T2_kernel_IF_like_flexible`.
- `dro_feature_selection/estimators.py`: conditional-mean and gradient estimators, plus compatibility exports for the kernel functions used by the paper experiments.
- `run_experiment.py`: unified, checkpointed experiment driver for synthetic, UCI, and ACS experiments.
- `dro_feature_selection/`: importable package containing data loading, checkpointing, feature selection, estimators, baselines, evaluation, and visualization.
- `dro_feature_selection/data/`: synthetic, UCI, ACS, and baseline-failure dataset implementations.
- `scripts/slurm/`: Bash/SLURM launchers for the paper experiment families.
- `scripts/analysis/`: result-processing utilities.
- `tests/`: fast characterization and regression tests that protect the paper implementation during refactors.

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

## Development

Run the characterization, data-pipeline, and entry-point tests with:

```bash
python -m unittest discover -s tests
```

The suite checks the kernel implementation, a deterministic paper-pipeline reference case, UCI and ACS preprocessing, classification baselines, checkpoint resume behavior, and the documented launchers. Tests involving downloaded datasets use small local fixtures in CI; full dataset runs download raw data into `datasets/` by default.

Repository refactors follow the scientific and reproducibility constraints in [`docs/refactoring-invariants.md`](docs/refactoring-invariants.md). These tests and invariants are intentionally version-controlled: they are the safeguards that make structural cleanup safe.
