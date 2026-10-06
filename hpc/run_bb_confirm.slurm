#!/bin/bash
# BB144 confirmation: exact ILP MLD oracle at 2500 shots per rate.
# Array 0-3: p in (0.02, 0.05, 0.08, 0.10). Confirms the 10^5-shot
# no-oracle bb144 findings on a smaller sample where every failing
# instance carries an exact oracle label. BP+OSD run on CUDA; the
# HiGHS MILP oracle stays on the CPU and is the runtime wall
# (roughly shots x fail-rate ILP solves per cell, cached per
# syndrome). Results land in results/bb144_confirm/, separate from
# the main results/hpc summary.
#
# Usage:
#   sbatch hpc/run_bb_confirm.slurm
#
# All 2500 shots per rate carry exact oracle labels; the LP optimal-face
# calibration stays capped at 300 distinct syndromes (one face costs
# ~14 s on bb144, 2n+1 LP solves each).

#SBATCH --job-name=fm_bb_conf
#SBATCH --output=hpc/logs/fm_bb_conf_%A_%a.out
#SBATCH --error=hpc/logs/fm_bb_conf_%A_%a.err
#SBATCH --array=0-3
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00

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

PS=(0.02 0.05 0.08 0.10)
P="${PS[$SLURM_ARRAY_TASK_ID]}"
SEED=$((1000 + SLURM_ARRAY_TASK_ID))
PTAG="${P/./_}"

echo "=== bb144 confirm p=${P} shots=2500 seed=${SEED} python=${PY} ==="
"${PY}" -u scripts/run_study.py \
    --codes bb144 \
    --p "${P}" \
    --shots 2500 \
    --device cuda \
    --lp-max-syndromes 300 \
    --ilp-time-limit 120 \
    --seed "${SEED}" \
    --out "results/bb144_confirm/bb144_p${PTAG}"
echo "=== bb144 confirm p=${P} finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
