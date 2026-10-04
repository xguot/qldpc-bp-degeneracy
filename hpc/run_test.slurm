#!/bin/bash
# Sanity job: full unit suite plus the exhaustive five-code study.
# Confirms the environment before the sweeps.
#
# Usage:
#   sbatch hpc/run_test.slurm

#SBATCH --job-name=fm_test
#SBATCH --output=hpc/logs/fm_test_%j.out
#SBATCH --error=hpc/logs/fm_test_%j.err
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p hpc/logs

PY="python3"
for CAND in \
    "${HOME}/miniforge3/envs/failuremodes/bin/python" \
    "${HOME}/.conda/envs/failuremodes/bin/python"; do
    if [[ -x "${CAND}" ]]; then
        PY="${CAND}"
        break
    fi
done

echo "=== python: ${PY} ==="
"${PY}" -m unittest discover -s tests -v
"${PY}" -u scripts/run_study.py --out results/hpc_smoke
echo "=== smoke study finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
