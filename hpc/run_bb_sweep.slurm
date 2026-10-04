#!/bin/bash
# Sampled failure-mode characterization on the bivariate bicycle codes
# without the MLD oracle: BP vs BP+OSD failure rates, convergence
# split, stopping-set counts, and the LP optimal-face calibration on
# sampled syndromes. The BP pass and the batched OSD run on CUDA
# (auto-detected), so the decoder stage takes 10^5 shots per task; the
# HiGHS LP face calibration stays on the CPU. Array 0-7: two codes
# (bb72, bb144) times four channel rates. Fixed seeds keep every run
# reproducible. bb144 gets a smaller LP cap because the face
# characterization costs 2n+1 LP solves per syndrome.
#
# Usage:
#   sbatch hpc/run_bb_sweep.slurm

#SBATCH --job-name=fm_bb
#SBATCH --output=hpc/logs/fm_bb_%A_%a.out
#SBATCH --error=hpc/logs/fm_bb_%A_%a.err
#SBATCH --array=0-7
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=8:00:00

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

CODES=(bb72 bb144)
PS=(0.02 0.05 0.08 0.10)
CODE="${CODES[$((SLURM_ARRAY_TASK_ID / 4))]}"
P="${PS[$((SLURM_ARRAY_TASK_ID % 4))]}"
SEED=$((2000 + SLURM_ARRAY_TASK_ID))
PTAG="${P/./_}"
CAP=300
if [[ "${CODE}" == "bb144" ]]; then
    CAP=100
fi

echo "=== ${CODE} p=${P} seed=${SEED} lp-cap=${CAP} python=${PY} ==="
"${PY}" -u scripts/run_study.py \
    --codes "${CODE}" \
    --p "${P}" \
    --shots 100000 \
    --no-oracle \
    --lp-max-syndromes "${CAP}" \
    --seed "${SEED}" \
    --out "results/hpc/${CODE}_p${PTAG}"
echo "=== ${CODE} p=${P} finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
