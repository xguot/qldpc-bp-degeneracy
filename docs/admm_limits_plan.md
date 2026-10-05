# ADMM limits under measurement noise - study plan

Working title: **On the Limits of Parity-Polytope and ADMM Decoders
Under Measurement Noise for Quantum LDPC Codes**.

Target: the [[144,12,12]] bivariate bicycle code with qudec's batched
parity-polytope ADMM decoder (Gu and Soleimanifar, ISIT 2026) under
phenomenological noise: d rounds of data errors and measurement
errors, both at rate p, decoded on the time-expanded matrix with OSD
post-processing (qudec `phenom.py` + `admm.py`).

## 1. Confirmation runs (2000+ shots)

Reference decoder (rho = 2, OSD-CS, plain ADMM), 4000 shots per cell,
grid d x p = {1, 3, 5} x {0.02, 0.05, 0.08, 0.10}. Establishes the
bb144 logical error rate versus rounds and rate under measurement
noise, with fixed seeds for reproducibility.

    sbatch hpc/run_admm_confirm.slurm     # 12 tasks

## 2. Robustness ablation

At d = 3, p = 0.05 and 0.08, 2500 shots per cell, sweep the ADMM
configuration to show no tuning rescues the decoder:

- penalty rho in {0.5, 1, 2, 4, 8} x OSD order {0, 1}   (10 configs)
- over-relaxation alpha in {1.5, 1.8} at rho = 2, OSD-CS (2 configs)
- LDR subgradient step beta in {0.5, 1, 2} at rho = 2, OSD-CS
  (3 configs)

    sbatch hpc/run_admm_ablate.slurm      # 30 tasks

## 3. Claims to test

- ADMM logical error rate under measurement noise degrades with d and
  p in a measurable, seed-stable way (confirmation grid).
- Within the sweep, LER is flat in rho, OSD order, alpha, and beta:
  the failure is a property of the parity-polytope relaxation under
  syndrome noise, not of ADMM's tuning.
- Where informative, compare against the same decoder on the exact
  (no measurement noise) syndrome to isolate the noise channel as the
  source of the limit.

## Results

Per-task JSONs land in results/admm_bb144/; aggregate with

    .venv/bin/python scripts/aggregate_admm.py

Potential follow-up axes: separate measurement rate p_meas from data
rate p, depolarizing instead of X-only noise, and the LDR variant on
the full grid.
