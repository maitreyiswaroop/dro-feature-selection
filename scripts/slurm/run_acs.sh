#!/bin/bash
set -euo pipefail
#SBATCH --job-name="acs_expt"
#SBATCH --output=./logs/acs_expt_%j.out
#SBATCH --error=./logs/acs_expt_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPOSITORY_ROOT"

seed=${SEED:-123}
SAVE_PATH="./results/acs/paper/seed_${seed}/"

bash "$SCRIPT_DIR/run_experiment.sh" \
  --populations acs \
  --acs-data-fraction 0.05 \
  --m1 10 \
  --m 18 \
  --dataset-size 30000 \
  --noise-scale 0.0 \
  --corr-strength 0.0 \
  --budget 15 \
  --learning-rate 0.01 \
  --penalty-type Reciprocal_L1 \
  --penalty-lambda 0.0001 \
  --optimizer-type adam \
  --parameterization alpha \
  --alpha-init random_5 \
  --num-epochs 100 \
  --patience 10 \
  --param-freezing \
  --gradient-mode autograd \
  --objective-value-estimator if \
  --t2-estimator-type mc_plugin \
  --N-grad-samples 10 \
  --no-use-baseline \
  --estimator-type plugin \
  --base-model-type xgb \
  --k-kernel 500 \
  --lasso-alpha 0.1 \
  --seed "$seed" \
  --save-path "$SAVE_PATH" \
  --force-regenerate-data
