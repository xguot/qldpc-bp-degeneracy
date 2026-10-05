# Failure-Mode Characterization of BP Decoding on Degenerate QLDPC Codes

## The claim

> Across the tested degenerate hypergraph-product codes ([[25,1,4]]) at code capacity with p = 0.1, 100.0% of avoidable BP+OSD failures are degenerate ambiguities, and a fractional LP optimum flags an avoidable BP+OSD failure with 21.4% precision (100.0% recall).

## Method

- X-error side only; the CSS sides decouple and the study codes are symmetric enough that the Z side is structurally identical.
- Sampled mode: 10000 i.i.d. shots per code with the fixed seed 1003; rates are empirical and every number below is reproducible.
- MLD oracle: integer programming (HiGHS MILP, one or two solves per query, cached per syndrome and logical class).
- An avoidable failure is a decode where the MLD oracle succeeds on the same syndrome but the decoder fails logically.
- Degenerate ambiguity: the syndrome has minimal errors in more than one logical class. Stopping set: every check adjacent to the true error support touches at least two flipped positions (the test on the residual is vacuous for syndrome-valid corrections).
- Wrong convergence: BP output satisfies all checks but is logically wrong; non-convergence: BP output violates the syndrome.
- The LP detector reports whether the optimal face of the relaxation admits a fractional point: the optimum value is fixed and each coordinate's range on that face is bounded with 2n extra LP solves per syndrome. A syndrome with two minimal errors always yields one (their midpoint is fractional). The vertex signal is also recorded.

## Codes under test

| code | params | n | k | d | m_x | m_z | syndromes | degenerate | ambiguous | instances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hgp44 | [[25,1,4]] | 25 | 1 | 4 | 12 | 12 | 2065 | n/a | n/a | 10000 |

## Experiment 1: avoidable-failure rates

| code | p | P(fail BP) | P(avoid BP) | P(fail BP+OSD) | P(avoid BP+OSD) | avoid share BP | avoid share BP+OSD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hgp44 | 0.1 | 0.1828 | 0.1297 | 0.1493 | 0.0885 | 0.710 | 0.593 |

## Experiment 2: classification of avoidable failures

### BP+OSD

| code | avoidable | ambiguous | stopping | both | neither |
| --- | --- | --- | --- | --- | --- |
| hgp44 | 885.0 | 885.0 | 0.0 | 0.0 | 0.0 |

### BP only (convergence split)

| code | avoidable | non-convergence | wrong convergence |
| --- | --- | --- | --- |
| hgp44 | 1297.0 | 1431 | 397 |

## Experiment 3: LP calibration as degeneracy detector

| code | solved | frac face | frac vertex | frac ambig | frac unambig | precision | recall | precision (u) | recall (u) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hgp44 | 300 | 211 | 47 | 118.0 | 93.0 | 0.214 | 1.000 | 0.214 | 1.000 |

## Findings

- Every avoidable BP+OSD failure on the degenerate codes is a degenerate ambiguity (885/885 on [[25,1,4]] (0 also stopping sets)); stopping sets add no separate failure mode for BP+OSD, they only co-occur with ambiguity.
- BP-only failures split by code: 1431 non-convergent and 397 wrongly convergent on [[25,1,4]].
- The LP optimal-face signal detects multi-minimal syndromes (211/300 sampled syndromes fractional on [[25,1,4]]); its precision below 100% comes from multi-minimal-but-unambiguous syndromes, i.e. degeneracy without ambiguity.

## Limitations

- Sampled rates are empirical (fixed seed) and confined to the X-error side; the ILP oracle is exact but subject to the per-solve time limit, and unresolved queries are dropped from the avoidable counts.
- Code distances are the analytic construction values, not brute-force verified at these sizes.
- The fractional-LP signal is a characterization of the optimal face via HiGHS; different solver paths may still disagree at the tolerance level.
- Degenerate-ambiguity labels depend on the logical basis that qudec.logicals returns.
