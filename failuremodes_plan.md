# Failure-Mode Characterization of BP Decoding on Degenerate Quantum LDPC Codes

## Goal

Measure why belief propagation (BP) fails on degenerate quantum LDPC codes,
and produce one quantified claim: how much of the avoidable failure comes
from degeneracy (multiple minimal errors per syndrome) versus classical
stopping sets, and whether the LP relaxation's fractional solutions predict
those failures. The deliverable is a small study with tables, tests, and a
single crisp claim. It is not a new decoder and not a race against the
state of the art.

## Assets to reuse

The sibling repository `../qudec` contains everything needed:

- `src/qudec/codes.py` — CSS code constructions (Steane, repetition,
  bivariate bicycle), GF(2) rank/RREF/nullspace, logical operator bases.
- `src/qudec/lp.py` — the exact LP relaxation of Gu-Soleimanifar
  (IEEE TIT 2026, Eq. 2), solved with HiGHS; `solve_lp(h, s)` returns the
  fractional solution for one syndrome.
- `src/qudec/bposd.py` — min-sum BP with OSD post-processing.
- `src/qudec/bench.py`, `src/qudec/noise.py` — i.i.d. sampling and the
  LER benchmark harness.
- `tests/test_bposd.py` — contains the brute-force MLD oracle pattern
  (`test_steane_close_to_mld`): enumerate all error patterns, map each
  syndrome to its minimal-weight error. Reuse this construction.

Work in a new repository that imports qudec (`pip install -e ../qudec`),
do not copy the decoder code.

## Background

Degeneracy is the known root cause of BP's poor behavior on quantum LDPC
codes (SymBreak arXiv:2412.02885; Gu and Soleimanifar characterize the LP
side). But the quantitative split — degenerate ambiguity versus stopping
sets versus plain non-convergence — is under-documented, and the LP's
fractional solutions are a natural degeneracy detector that nobody has
calibrated. That calibration is the study.

## Experiments, in order

1. **Oracle and failure set.** On small codes (Steane [[7,1,3]],
   repetition, plus constructed degenerate CSS codes with n <= 20 via the
   GF(2) tools), build the syndrome-to-minimal-error oracle by brute force.
   Run BP and BP+OSD; a failure is "avoidable" when the oracle decodes the
   same syndrome successfully. Report avoidable-failure rates per code and
   per error weight.

2. **Classify every avoidable failure** into:
   - **Degenerate ambiguity**: the syndrome has more than one minimal
     error with inequivalent logical outcomes. Measure the count via the
     oracle, and separately measure the LP fractional-solution rate for
     the same syndrome (fraction of fractional optima, or the count of
     non-integral coordinates).
   - **Stopping set**: the flipped positions form a stopping set in the
     Tanner graph (every adjacent check touches at least two flipped
     positions). Check this on the residual pattern.
   - **Non-convergence vs wrong convergence**: did BP satisfy all checks
     and stop at the wrong answer, or run out of iterations unsatisfied?

3. **Calibrate the LP as a degeneracy detector.** For each syndrome,
   record whether the LP optimum is fractional and whether decoding it is
   ambiguous. Report precision/recall of "fractional LP implies avoidable
   BP failure" per code family.

4. **Write the claim.** One sentence, e.g. "on the tested degenerate
   codes, X percent of avoidable BP+OSD failures are degenerate
   ambiguities, and LP fractional solutions flag them with Y percent
   precision." Everything else supports that sentence.

## Milestones

- Week 1: oracle plus classifier on Steane and repetition, with unit tests
  for each classifier (synthetic syndromes with known labels).
- Week 2: two constructed degenerate codes (hypergraph products of small
  classical codes, n <= 20), full failure tables.
- Week 3: writeup with the claim, the tables, and the honest limitations
  (small code sizes, code capacity only).

## Rules

- Every classifier gets a unit test against hand-constructed cases.
- All random experiments use fixed seeds and are reproducible.
- Report BP failures and BP+OSD failures separately; never conflate them.
- Do not benchmark against state-of-the-art decoders and do not expand
  scope to circuit-level noise. The study is the classification, not the
  competition.
