#!/bin/bash
set -euo pipefail
#SBATCH --job-name="dro_feature_selection"
#SBATCH --output=./logs/dro_feature_selection_%j.out
#SBATCH --error=./logs/dro_feature_selection_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4  # Increased CPU count for CPU-only processing
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Loading environment..."
source "/home/$USER/miniconda/etc/profile.d/conda.sh"
conda activate venv   # or your env name

# # Force PyTorch to use CPU only
# export CUDA_VISIBLE_DEVICES=""
# export PYTORCH_DEVICE="cpu"
# export VSS_USE_CPU_FALLBACK=1  # Your custom env variable if implemented
# # Set PyTorch to use all available CPUs
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPOSITORY_ROOT"
python3 "$REPOSITORY_ROOT/run_experiment.py" "$@"
