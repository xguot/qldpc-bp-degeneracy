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
- `src/failuremodes/gpu_osd.py` — batched, bit-packed GF(2) OSD
  elimination in torch: swap-free elimination with the same pivot
  choices as qudec's CPU `_osd`, vectorized across shots, runs on CUDA.
- `src/failuremodes/gpu_decoders.py` — `GpuBpOsdDecoder` and
  `GpuBpOnlyDecoder`: qudec's BP pass fed CUDA tensors plus the batched
  GPU OSD, chunked to bound VRAM. Output equals the CPU decoder exactly
  (unit-tested shot for shot).
- `src/failuremodes/study.py` — the experiment harness (experiments
  1-3 of the plan): exhaustive and sampled paths over a code registry.
- `src/failuremodes/report.py` — claim generation and markdown report
  rendering.
- `scripts/run_study.py` — CLI for one run; `scripts/aggregate_hpc.py`
  merges per-run results into one summary table;
  `scripts/make_figures.py` renders the note figures.
- `docs/` — the LaTeX research note (`note.pdf`) with the claim, the
  lemma, the tables, and the limitations.
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
- The GPU decoders reproduce the CPU decoder output exactly: the
  batched OSD uses the same pivot choices and combination sweep as
  qudec's `_osd`, and the equality is unit-tested on random inputs.
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

The decoder stage (batched BP + OSD) runs on CUDA with the job
configs from the `lensless-recon` allocation (`nssac_students`,
`bii-gpu`, one GPU, 8 CPUs, 64 GB); the HiGHS LP/MILP stages stay on
the CPU. The environment installs a CUDA-capable torch. Push the code,
set up the environment once (installs only, nothing runs on the login
node), and submit the sweeps:

    bash hpc/sync.sh push
    ssh rivanna
    cd ~/scratch/qldpc-bp-degeneracy
    bash hpc/setup.sh                          # conda env failuremodes + qudec
    sbatch hpc/run_hgp_sweep.slurm             # [[25,1,4]] [[41,1,5]] [[61,1,6]] x 4 p, 10^4 shots, ILP oracle
    sbatch hpc/run_bb_sweep.slurm              # [[72,12,6]] [[144,12,12]] x 4 p, 10^5 shots, GPU decoders

Pull results back and aggregate:

    bash hpc/sync.sh pull
    .venv/bin/python scripts/aggregate_hpc.py  # results/hpc/hpc_summary.md

`hpc/setup.sh` expects the qudec repo at `~/scratch/qudec` (override
with `QUDEC_PATH`).

## Scaled findings (Rivanna, 20 runs, committed in results/hpc)

HGP surface patches [[25,1,4]], [[41,1,5]], [[61,1,6]] at
p = 0.02-0.10 with the exact ILP oracle, and [[72,12,6]], [[144,12,12]]
at 10^5 shots without it:

- 98-100% of avoidable BP+OSD failures on the surface patches are
  degenerate ambiguities at every p; stopping sets contribute zero
  avoidable failures anywhere.
- The LP optimal-face detector keeps 100% recall on the patches while
  its precision falls with distance: 47.6% at d <= 3 (exhaustive),
  21% at d = 4, 9% at d = 5-6. The signal tracks degeneracy, not
  failure: large codes accumulate harmless degenerate syndromes.
- About half of BP+OSD failures on the patches are avoidable
  (MLD-recoverable) at p = 0.1; the rest are true MLD limits.
- BP-only failures are dominated by non-convergence, and its share
  grows with distance (78% at d = 4 to 96% at d = 6 at p = 0.1); on
  [[72,12,6]] roughly 40% of BP failures are wrong convergences, while
  [[144,12,12]] almost never wrong-converges (3%).
- On the bicycle codes the LP detector's precision rises with p
  ([[72,12,6]]: 41% at p = 0.02 to 86% at p = 0.1) with recall
  90-100%.

## The note

`docs/note.pdf` is a six-page research note: claim, setup, the lemma
(ambiguous syndrome implies fractional LP optimal face), the measured
tables, and honest limitations. Regenerate with

    .venv/bin/python scripts/make_figures.py
    make -C docs

## Claim

> Across the tested degenerate hypergraph-product codes ([[5,1,2]],
> [[8,1,2]], [[13,1,3]]) at code capacity with p = 0.1, 100.0% of
> avoidable BP+OSD failures are degenerate ambiguities, and a
> fractional LP optimum flags an avoidable BP+OSD failure with 47.6%
> precision (100.0% recall); the non-degenerate [[7,1,3]] Steane code
> has no avoidable BP+OSD failures at this p.

Full tables, per-weight breakdowns, and limitations: `results/REPORT.md`.
The claim updates automatically as HPC results land.
