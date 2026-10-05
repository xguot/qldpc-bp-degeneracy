#!/bin/bash
# Sampled failure-mode sweep on the larger hypergraph-product surface
# patches with the exact ILP MLD oracle. Array 0-11: three codes
# (hgp44, hgp55, hgp66) times four channel rates. Fixed seeds keep
# every run reproducible. The BP pass and the batched OSD run on CUDA
# (auto-detected); the HiGHS MILP oracle and LP calibration stay on the
# CPU, and the MILP oracle is the runtime wall, which is why the HGP
# sweep keeps 10^4 shots while the BB sweep takes 10^5.
#
# Usage:
#   sbatch hpc/run_hgp_sweep.slurm

#SBATCH --job-name=fm_hgp
#SBATCH --output=hpc/logs/fm_hgp_%A_%a.out
#SBATCH --error=hpc/logs/fm_hgp_%A_%a.err
#SBATCH --array=0-11
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00

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

CODES=(hgp44 hgp55 hgp66)
PS=(0.02 0.05 0.08 0.10)
CODE="${CODES[$((SLURM_ARRAY_TASK_ID / 4))]}"
P="${PS[$((SLURM_ARRAY_TASK_ID % 4))]}"
SEED=$((1000 + SLURM_ARRAY_TASK_ID))
PTAG="${P/./_}"

echo "=== ${CODE} p=${P} seed=${SEED} python=${PY} ==="
"${PY}" -u scripts/run_study.py \
    --codes "${CODE}" \
    --p "${P}" \
    --shots 10000 \
    --device cuda \
    --lp-max-syndromes 300 \
    --ilp-time-limit 120 \
    --seed "${SEED}" \
    --out "results/hpc/${CODE}_p${PTAG}"
echo "=== ${CODE} p=${P} finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
