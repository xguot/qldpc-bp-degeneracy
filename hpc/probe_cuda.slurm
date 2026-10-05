#!/bin/bash
# Probe a GPU node for torch/CUDA status before the sweeps. Tiny job.
# Exits nonzero when CUDA is unavailable, so success is visible from
# the job status alone. Login nodes have no GPU, which is why
# torch.cuda.is_available() is False there even with a CUDA build.
#
# Usage:
#   sbatch hpc/probe_cuda.slurm

#SBATCH --job-name=fm_probe
#SBATCH --output=hpc/logs/fm_probe_%j.out
#SBATCH --error=hpc/logs/fm_probe_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|a6000|a40"
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0:10:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT_DIR}"
mkdir -p hpc/logs

nvidia-smi
echo "---"

PY="python3"
for CAND in \
    "${HOME}/miniforge3/envs/failuremodes/bin/python" \
    "${HOME}/.conda/envs/failuremodes/bin/python"; do
    if [[ -x "${CAND}" ]]; then
        PY="${CAND}"
        break
    fi
done

"${PY}" -c "
import sys, torch
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
print('device', torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device='cuda')
print('matmul check', float((x @ x).sum()))
"
