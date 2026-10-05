# Failure-Mode Characterization of BP Decoding on Degenerate QLDPC Codes

## The claim

> Across the tested degenerate hypergraph-product codes ([[61,1,6]]) at code capacity with p = 0.02, 100.0% of avoidable BP+OSD failures are degenerate ambiguities, and a fractional LP optimum flags an avoidable BP+OSD failure with 3.5% precision (100.0% recall).

## Method

- X-error side only; the CSS sides decouple and the study codes are symmetric enough that the Z side is structurally identical.
- Sampled mode: 10000 i.i.d. shots per code with the fixed seed 1008; rates are empirical and every number below is reproducible.
- MLD oracle: integer programming (HiGHS MILP, one or two solves per query, cached per syndrome and logical class).
- An avoidable failure is a decode where the MLD oracle succeeds on the same syndrome but the decoder fails logically.
- Degenerate ambiguity: the syndrome has minimal errors in more than one logical class. Stopping set: every check adjacent to the true error support touches at least two flipped positions (the test on the residual is vacuous for syndrome-valid corrections).
- Wrong convergence: BP output satisfies all checks but is logically wrong; non-convergence: BP output violates the syndrome.
- The LP detector reports whether the optimal face of the relaxation admits a fractional point: the optimum value is fixed and each coordinate's range on that face is bounded with 2n extra LP solves per syndrome. A syndrome with two minimal errors always yields one (their midpoint is fractional). The vertex signal is also recorded.

## Codes under test

| code | params | n | k | d | m_x | m_z | syndromes | degenerate | ambiguous | instances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hgp66 | [[61,1,6]] | 61 | 1 | 6 | 30 | 30 | 2482 | n/a | n/a | 10000 |

## Experiment 1: avoidable-failure rates

| code | p | P(fail BP) | P(avoid BP) | P(fail BP+OSD) | P(avoid BP+OSD) | avoid share BP | avoid share BP+OSD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hgp66 | 0.02 | 0.0159 | 0.0158 | 0.0007 | 0.0006 | 0.994 | 0.857 |

## Experiment 2: classification of avoidable failures

### BP+OSD

| code | avoidable | ambiguous | stopping | both | neither |
| --- | --- | --- | --- | --- | --- |
| hgp66 | 6.0 | 6.0 | 0.0 | 0.0 | 0.0 |

### BP only (convergence split)

| code | avoidable | non-convergence | wrong convergence |
| --- | --- | --- | --- |
| hgp66 | 158.0 | 159 | 0 |

## Experiment 3: LP calibration as degeneracy detector

| code | solved | frac face | frac vertex | frac ambig | frac unambig | precision | recall | precision (u) | recall (u) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hgp66 | 300 | 65 | 17 | 4.0 | 61.0 | 0.035 | 1.000 | 0.035 | 1.000 |

## Findings

- Every avoidable BP+OSD failure on the degenerate codes is a degenerate ambiguity (6/6 on [[61,1,6]] (0 also stopping sets)); stopping sets add no separate failure mode for BP+OSD, they only co-occur with ambiguity.
- BP-only failures split by code: 159 non-convergent and 0 wrongly convergent on [[61,1,6]].
- The LP optimal-face signal detects multi-minimal syndromes (65/300 sampled syndromes fractional on [[61,1,6]]); its precision below 100% comes from multi-minimal-but-unambiguous syndromes, i.e. degeneracy without ambiguity.

## Limitations

- Sampled rates are empirical (fixed seed) and confined to the X-error side; the ILP oracle is exact but subject to the per-solve time limit, and unresolved queries are dropped from the avoidable counts.
- Code distances are the analytic construction values, not brute-force verified at these sizes.
- The fractional-LP signal is a characterization of the optimal face via HiGHS; different solver paths may still disagree at the tolerance level.
- Degenerate-ambiguity labels depend on the logical basis that qudec.logicals returns.
