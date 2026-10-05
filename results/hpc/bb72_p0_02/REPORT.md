# Failure-Mode Characterization of BP Decoding on Degenerate QLDPC Codes

## The claim

> Across the tested bivariate bicycle codes ([[72,12,6]]) at code capacity with p = 0.02, a fractional LP optimum flags a BP+OSD failure with 40.7% precision (100.0% recall); BP-only failures split into 52.8% non-convergence and 47.2% wrong convergence.

## Method

- X-error side only; the CSS sides decouple and the study codes are symmetric enough that the Z side is structurally identical.
- Sampled mode: 100000 i.i.d. shots per code with the fixed seed 2000; rates are empirical and every number below is reproducible.
- MLD oracle: integer programming (HiGHS MILP, one or two solves per query, cached per syndrome and logical class).
- An avoidable failure is a decode where the MLD oracle succeeds on the same syndrome but the decoder fails logically.
- Degenerate ambiguity: the syndrome has minimal errors in more than one logical class. Stopping set: every check adjacent to the true error support touches at least two flipped positions (the test on the residual is vacuous for syndrome-valid corrections).
- Wrong convergence: BP output satisfies all checks but is logically wrong; non-convergence: BP output violates the syndrome.
- The LP detector reports whether the optimal face of the relaxation admits a fractional point: the optimum value is fixed and each coordinate's range on that face is bounded with 2n extra LP solves per syndrome. A syndrome with two minimal errors always yields one (their midpoint is fractional). The vertex signal is also recorded.

## Codes under test

| code | params | n | k | d | m_x | m_z | syndromes | degenerate | ambiguous | instances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bb72 | [[72,12,6]] | 72 | 12 | 6 | 36 | 36 | 18911 | n/a | n/a | 100000 |

## Experiment 1: avoidable-failure rates

| code | p | P(fail BP) | P(avoid BP) | P(fail BP+OSD) | P(avoid BP+OSD) | avoid share BP | avoid share BP+OSD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bb72 | 0.02 | 0.0135 | n/a | 0.0106 | n/a | n/a | n/a |

## Experiment 2: classification of avoidable failures

### BP+OSD

| code | avoidable | ambiguous | stopping | both | neither |
| --- | --- | --- | --- | --- | --- |
| bb72 | n/a | n/a | 0.0 | n/a | n/a |

### BP only (convergence split)

| code | avoidable | non-convergence | wrong convergence |
| --- | --- | --- | --- |
| bb72 | n/a | 712 | 636 |

## Experiment 3: LP calibration as degeneracy detector

| code | solved | frac face | frac vertex | frac ambig | frac unambig | precision | recall | precision (u) | recall (u) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bb72 | 300 | 26 | 2 | n/a | n/a | 0.407 | 1.000 | 0.407 | 1.000 |

## Findings

- On [[72,12,6]] the LP optimal face is fractional for 26/300 sampled syndromes (2 by the vertex signal) and flags BP+OSD failures with 0.407 precision at 1.000 recall (p = 0.02).

## Limitations

- Sampled rates are empirical (fixed seed) and confined to the X-error side; the ILP oracle is exact but subject to the per-solve time limit, and unresolved queries are dropped from the avoidable counts.
- Code distances are the analytic construction values, not brute-force verified at these sizes.
- The fractional-LP signal is a characterization of the optimal face via HiGHS; different solver paths may still disagree at the tolerance level.
- Degenerate-ambiguity labels depend on the logical basis that qudec.logicals returns.
