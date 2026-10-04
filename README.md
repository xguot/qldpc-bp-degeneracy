# failuremodes

Failure-mode characterization of belief propagation (BP) decoding on
degenerate quantum LDPC codes. Measures how much of the avoidable BP
failure comes from degeneracy (multiple minimal errors per syndrome)
versus classical stopping sets, and calibrates the LP relaxation's
fractional solutions as a degeneracy detector. The study design lives
in `failuremodes_plan.md`; the measured claim and tables are in
`results/REPORT.md`.

Two scales:

- **Exhaustive** (local): the five small codes (Steane, repetition,
  HGP patches [[5,1,2]], [[8,1,2]], [[13,1,3]]) over all 2^n error
  patterns — every number is exact.
- **Sampled** (HPC): larger surface patches ([[25,1,4]] ... [[85,1,7]])
  and the bivariate bicycle codes [[72,12,6]], [[144,12,12]] with
  i.i.d. shots, fixed seeds, and an exact MLD oracle solved as an
  integer program (HiGHS MILP).

## Layout

- `src/failuremodes/oracle.py` — brute-force MLD oracle: per syndrome,
  all minimal-weight errors and their logical classes (ambiguity), plus
  brute-force distance.
- `src/failuremodes/ilp_oracle.py` — exact MLD oracle via integer
  programming (one or two MILP solves per query, cached), for codes too
  large to enumerate. Validated against the brute-force oracle in the
  test suite.
- `src/failuremodes/constructed.py` — hypergraph product of two
  classical codes; `hgp_repetition(d1, d2)` builds the surface-code
  patches.
- `src/failuremodes/classify.py` — stopping-set test, syndrome
  satisfaction, convergence label, logical success.
- `src/failuremodes/lp_detector.py` — fractional-LP detector on
  qudec's exact LP (`solve_lp`); `lp_face_fractional` characterizes the
  whole optimal face, not just one solver vertex.
- `src/failuremodes/bp_only.py` — `BpOnlyDecoder`, qudec min-sum BP
  with hard decisions and no OSD stage, so non-convergence stays
  observable.
- `src/failuremodes/study.py` — the experiment harness (experiments
  1-3 of the plan): exhaustive and sampled paths over a code registry.
- `src/failuremodes/report.py` — claim generation and markdown report
  rendering.
- `scripts/run_study.py` — CLI for one run; `scripts/aggregate_hpc.py`
  merges per-run results into one summary table.
- `hpc/` — Rivanna sync, one-time setup, and the SLURM sweeps.
- `tests/` — unit tests, including hand-constructed cases with known
  labels for every classifier and exact ILP-vs-brute-force agreement.

## Design decisions

- Study runs on the X-error side only: CSS sides decouple, and each
  side uses its own check/logical pair.
- Exhaustive mode: every number is an exact count over all 2^n error
  patterns, no sampling, no randomness. Sampled mode: i.i.d. shots with
  a fixed seed; every run is reproducible.
- Avoidable failure: the MLD oracle decodes the syndrome successfully
  but the decoder fails logically. BP and BP+OSD are always reported
  separately.
- Stopping set: every check adjacent to the true error support touches
  at least two flipped positions. Applied to the true error support
  because the test on a syndrome-valid residual is vacuous (any
  codeword support is trivially a stopping set).
- Degenerate ambiguity: the syndrome has minimal errors in more than
  one logical class.
- The LP detector fixes the optimum value and bounds each coordinate's
  range on the optimal face (2n extra LP solves per syndrome). A
  syndrome with two minimal errors always yields a fractional point
  (their midpoint), so ambiguous syndromes are always detected; the
  single HiGHS vertex would miss most of them.
- Degenerate-ambiguity labels depend on the logical basis returned by
  `qudec.logicals`.

## Install (offline, uses the sibling qudec repo)

    python3 -m venv --system-site-packages .venv
    .venv/bin/python -m pip install -e ../qudec --no-build-isolation --no-deps
    .venv/bin/python -m pip install -e . --no-build-isolation --no-deps

## Test and run

    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python scripts/run_study.py                 # exhaustive five-code study
    .venv/bin/python scripts/run_study.py --codes hgp44 --p 0.05 --shots 10000 --seed 1
    .venv/bin/python scripts/run_study.py --codes bb72 --p 0.08 --shots 10000 --no-oracle

## HPC (Rivanna)

The study is CPU-bound (torch on CPU, HiGHS LP/MILP), but the job
templates follow the `lensless-recon` allocation config
(`nssac_students`, `bii-gpu`). Push the code, set up the environment
once, and submit the sweeps:

    bash hpc/sync.sh push
    ssh rivanna
    cd ~/scratch/qldpc-bp-degeneracy
    bash hpc/setup.sh                          # conda env failuremodes + qudec
    sbatch hpc/run_test.slurm                  # unit suite + exhaustive smoke
    sbatch hpc/run_hgp_sweep.slurm             # [[25,1,4]] [[41,1,5]] [[61,1,6]] x 4 p, ILP oracle
    sbatch hpc/run_bb_sweep.slurm              # [[72,12,6]] [[144,12,12]] x 4 p, no oracle

Pull results back and aggregate:

    bash hpc/sync.sh pull
    .venv/bin/python scripts/aggregate_hpc.py  # results/hpc_summary.md

`hpc/setup.sh` expects the qudec repo at `~/scratch/qudec` (override
with `QUDEC_PATH`).

## Claim

> Across the tested degenerate hypergraph-product codes ([[5,1,2]],
> [[8,1,2]], [[13,1,3]]) at code capacity with p = 0.1, 100.0% of
> avoidable BP+OSD failures are degenerate ambiguities, and a
> fractional LP optimum flags an avoidable BP+OSD failure with 47.6%
> precision (100.0% recall); the non-degenerate [[7,1,3]] Steane code
> has no avoidable BP+OSD failures at this p.

Full tables, per-weight breakdowns, and limitations: `results/REPORT.md`.
The claim updates automatically as HPC results land.
