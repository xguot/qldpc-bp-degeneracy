#!/bin/bash
# One-time environment setup for the failure-mode study on Rivanna.
#
# Usage (on the login node, after `bash hpc/sync.sh push` and after the
# qudec repo is set up at ~/scratch/qudec, override with QUDEC_PATH):
#   bash hpc/setup.sh
#
# Creates a conda environment with CUDA-capable torch (the BP pass and
# the batched OSD run on the GPU), the qudec dependency, and the
# failuremodes package. Only installs; no jobs or tests run here.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

module purge
module load miniforge

ENV_NAME="failuremodes"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Conda environment '${ENV_NAME}' already exists."
else
    echo "=== Creating conda environment '${ENV_NAME}' ==="
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

QUDEC_PATH="${QUDEC_PATH:-${HOME}/scratch/qudec}"

echo "=== Installing Python dependencies ==="
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install numpy scipy
# Force the CUDA build: pip skips same-version wheel swaps, which is
# how a CPU-only torch survives an upgrade.
conda run -n "${ENV_NAME}" pip uninstall -y torch
conda run -n "${ENV_NAME}" pip install torch
conda run -n "${ENV_NAME}" pip install -e "${QUDEC_PATH}"
conda run -n "${ENV_NAME}" pip install -e .

echo ""
echo "=== Setup complete ==="
echo "Verify CUDA:    conda run -n ${ENV_NAME} python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "HGP sweep:       sbatch hpc/run_hgp_sweep.slurm"
echo "BB sweep:        sbatch hpc/run_bb_sweep.slurm"
echo "Monitor:         squeue -u \$USER"
echo "Pull results:    bash hpc/sync.sh pull"
