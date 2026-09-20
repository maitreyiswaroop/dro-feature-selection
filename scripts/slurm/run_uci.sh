#!/bin/bash
set -euo pipefail
#SBATCH --job-name="uci_expt"
#SBATCH --output=./logs/uci_expt_%j.out
#SBATCH --error=./logs/uci_expt_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPOSITORY_ROOT"

seed=${SEED:-2}

SAVE_PATH="./results/uci/paper/seed_${seed}/"
mkdir -p "$SAVE_PATH"

bash "$SCRIPT_DIR/run_experiment.sh" \
  --populations uci uci \
  --uci-populations Male Female \
  --uci-data-fraction 1.0 \
  --budget 10 \
  --penalty-type Reciprocal_L1 \
  --penalty-lambda 0.000005 \
  --learning-rate 0.02 \
  --optimizer-type adam \
  --parameterization theta \
  --alpha-init random_5 \
  --num-epochs 120 \
  --patience 15 \
  --param-freezing \
  --gradient-mode autograd \
  --t2-estimator-type mc_plugin \
  --N-grad-samples 25 \
  --no-use-baseline \
  --estimator-type plugin \
  --base-model-type xgb \
  --objective-value-estimator if \
  --k-kernel 500 \
  --lasso-alpha 0.1 \
  --seed "$seed" \
  --save-path "$SAVE_PATH" \
  --force-regenerate-data \
  --verbose
