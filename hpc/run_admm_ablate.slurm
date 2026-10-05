#!/bin/bash
# ADMM robustness ablation on [[144,12,12]] under phenomenological
# noise at d = 3 rounds. Array 0-29: 15 decoder configurations times
# two channel rates (p = 0.05, 0.08) at 2500 shots per cell.
#
# Local task id (id % 15):
#   0-9   rho grid {0.5, 1, 2, 4, 8} x OSD order {0, 1}
#   10-11 over-relaxation alpha {1.5, 1.8} at rho = 2, OSD-CS
#   12-14 LDR subgradient step beta {0.5, 1, 2} at rho = 2, OSD-CS
#
# The claim this supports: no setting of rho, OSD order, or step size
# rescues ADMM under measurement noise. Fixed seeds keep the grid
# reproducible.
#
# Usage:
#   sbatch hpc/run_admm_ablate.slurm

#SBATCH --job-name=fm_admm_a
#SBATCH --output=hpc/logs/fm_admm_a_%A_%a.out
#SBATCH --error=hpc/logs/fm_admm_a_%A_%a.err
#SBATCH --array=0-29
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100|a6000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00

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

LOCAL=$((SLURM_ARRAY_TASK_ID % 15))
if [[ "${SLURM_ARRAY_TASK_ID}" -lt 15 ]]; then
    P="0.05"
else
    P="0.08"
fi
SEED=$((1000 + SLURM_ARRAY_TASK_ID))

RHO="2.0"
OSD="1"
ALPHA="1.0"
LDR="0"
BETA="1.0"
if [[ "${LOCAL}" -lt 10 ]]; then
    RHOS=(0.5 1.0 2.0 4.0 8.0)
    RHO="${RHOS[$((LOCAL / 2))]}"
    OSD="$((LOCAL % 2))"
elif [[ "${LOCAL}" -lt 12 ]]; then
    ALPHAS=(1.5 1.8)
    ALPHA="${ALPHAS[$((LOCAL - 10))]}"
else
    LDR="1"
    BETAS=(0.5 1.0 2.0)
    BETA="${BETAS[$((LOCAL - 12))]}"
fi

echo "=== admm ablate bb144 d=3 p=${P} rho=${RHO} osd=${OSD} alpha=${ALPHA} ldr=${LDR} beta=${BETA} seed=${SEED} python=${PY} ==="
"${PY}" -u scripts/run_admm_study.py \
    --mode ablate \
    --config-id "${SLURM_ARRAY_TASK_ID}" \
    --code bb144 \
    --d 3 \
    --p "${P}" \
    --shots 2500 \
    --rho "${RHO}" \
    --osd-order "${OSD}" \
    --alpha "${ALPHA}" \
    --ldr "${LDR}" \
    --ldr-beta "${BETA}" \
    --seed "${SEED}"
echo "=== admm ablate bb144 p=${P} rho=${RHO} osd=${OSD} alpha=${ALPHA} ldr=${LDR} finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
