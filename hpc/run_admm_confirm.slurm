#!/bin/bash
# ADMM confirmation runs on [[144,12,12]] under phenomenological noise.
# Array 0-11: three round counts (d = 1, 3, 5) times four channel rates
# at 4000 shots per cell with the reference decoder (rho = 2, OSD-CS,
# plain ADMM). Data errors and measurement errors both at rate p.
# The ADMM relaxation runs on CUDA; the per-shot OSD elimination is
# CPU numpy, so d = 5 cells dominate the wall time.
#
# Usage:
#   sbatch hpc/run_admm_confirm.slurm

#SBATCH --job-name=fm_admm_c
#SBATCH --output=hpc/logs/fm_admm_c_%A_%a.out
#SBATCH --error=hpc/logs/fm_admm_c_%A_%a.err
#SBATCH --array=0-11
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p hpc/logs results/admm_bb144

PY="python3"
for CAND in \
    "${HOME}/miniforge3/envs/failuremodes/bin/python" \
    "${HOME}/.conda/envs/failuremodes/bin/python"; do
    if [[ -x "${CAND}" ]]; then
        PY="${CAND}"
        break
    fi
done

DS=(1 3 5)
PS=(0.02 0.05 0.08 0.10)
D="${DS[$((SLURM_ARRAY_TASK_ID / 4))]}"
P="${PS[$((SLURM_ARRAY_TASK_ID % 4))]}"
SEED=$((1000 + SLURM_ARRAY_TASK_ID))

echo "=== admm confirm bb144 d=${D} p=${P} shots=4000 seed=${SEED} python=${PY} ==="
"${PY}" -u scripts/run_admm_study.py \
    --mode confirm \
    --config-id "${SLURM_ARRAY_TASK_ID}" \
    --code bb144 \
    --d "${D}" \
    --p "${P}" \
    --shots 4000 \
    --rho 2.0 \
    --osd-order 1 \
    --alpha 1.0 \
    --seed "${SEED}"
echo "=== admm confirm bb144 d=${D} p=${P} finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
