# Failure-Mode Characterization of BP Decoding on Degenerate QLDPC Codes

## The claim

> Across the three degenerate hypergraph-product codes ([[5,1,2]], [[8,1,2]], [[13,1,3]]) at code capacity with p = 0.1, 100.0% of avoidable BP+OSD failures are degenerate ambiguities, and a fractional LP optimum flags an avoidable BP+OSD failure with 47.6% precision (100.0% recall); the non-degenerate [[7,1,3]] Steane code has no avoidable BP+OSD failures at this p.

## Method

- X-error side only; the CSS sides decouple and the study codes are symmetric enough that the Z side is structurally identical.
- Exhaustive enumeration of all 2^n error patterns per code (n <= 13): every rate below is exact, no sampling.
- An avoidable failure is a decode where the MLD oracle succeeds on the same syndrome but the decoder fails logically.
- Degenerate ambiguity: the syndrome has minimal errors in more than one logical class. Stopping set: every check adjacent to the true error support touches at least two flipped positions (the test on the residual is vacuous for syndrome-valid corrections).
- Wrong convergence: BP output satisfies all checks but is logically wrong; non-convergence: BP output violates the syndrome.
- The LP detector reports whether the optimal face of the relaxation admits a fractional point: the optimum value is fixed and each coordinate's range on that face is bounded with 2n extra LP solves per syndrome. A syndrome with two minimal errors always yields one (their midpoint is fractional). The vertex signal is also recorded.

## Codes under test

| code | params | n | k | d | m_x | m_z | syndromes | degenerate | ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steane | [[7,1,3]] | 7 | 1 | 3 | 3 | 3 | 8 | 0 | 0 |
| rep5 | [[5,1,5]] | 5 | 1 | 5 | 0 | 4 | 16 | 0 | 0 |
| hgp22 | [[5,1,2]] | 5 | 1 | 2 | 2 | 2 | 4 | 2 | 2 |
| hgp32 | [[8,1,2]] | 8 | 1 | 2 | 4 | 3 | 8 | 5 | 5 |
| hgp33 | [[13,1,3]] | 13 | 1 | 3 | 6 | 6 | 64 | 33 | 26 |

## Experiment 1: avoidable-failure rates

| code | p | P(fail BP) | P(avoid BP) | P(fail BP+OSD) | P(avoid BP+OSD) | avoid share BP | avoid share BP+OSD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| steane | 0.05 | 0.0415 | 0.0000 | 0.0415 | 0.0000 | 0.000 | 0.000 |
| steane | 0.1 | 0.1306 | 0.0000 | 0.1306 | 0.0000 | 0.000 | 0.000 |
| rep5 | 0.05 | 0.0012 | 0.0000 | 0.0012 | 0.0000 | 0.000 | 0.000 |
| rep5 | 0.1 | 0.0086 | 0.0000 | 0.0086 | 0.0000 | 0.000 | 0.000 |
| hgp22 | 0.05 | 0.0950 | 0.0860 | 0.0950 | 0.0860 | 0.905 | 0.905 |
| hgp22 | 0.1 | 0.1800 | 0.1476 | 0.1800 | 0.1476 | 0.820 | 0.820 |
| hgp32 | 0.05 | 0.1355 | 0.1212 | 0.1355 | 0.1212 | 0.895 | 0.895 |
| hgp32 | 0.1 | 0.2440 | 0.1978 | 0.2440 | 0.1978 | 0.811 | 0.811 |
| hgp33 | 0.05 | 0.0518 | 0.0313 | 0.0491 | 0.0283 | 0.603 | 0.577 |
| hgp33 | 0.1 | 0.1576 | 0.0869 | 0.1520 | 0.0803 | 0.552 | 0.529 |

### Per error weight (exact counts over all 2^n patterns)

**steane**

| w | patterns | fail BP | avoid BP | fail BP+OSD | avoid BP+OSD |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 7 | 0 | 0 | 0 | 0 |
| 2 | 21 | 21 | 0 | 21 | 0 |
| 3 | 35 | 7 | 0 | 7 | 0 |
| 4 | 35 | 28 | 0 | 28 | 0 |
| 5 | 21 | 0 | 0 | 0 | 0 |
| 6 | 7 | 7 | 0 | 7 | 0 |
| 7 | 1 | 1 | 0 | 1 | 0 |

**rep5**

| w | patterns | fail BP | avoid BP | fail BP+OSD | avoid BP+OSD |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 5 | 0 | 0 | 0 | 0 |
| 2 | 10 | 0 | 0 | 0 | 0 |
| 3 | 10 | 10 | 0 | 10 | 0 |
| 4 | 5 | 5 | 0 | 5 | 0 |
| 5 | 1 | 1 | 0 | 1 | 0 |

**hgp22**

| w | patterns | fail BP | avoid BP | fail BP+OSD | avoid BP+OSD |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 5 | 2 | 2 | 2 | 2 |
| 2 | 10 | 6 | 2 | 6 | 2 |
| 3 | 10 | 6 | 2 | 6 | 2 |
| 4 | 5 | 2 | 2 | 2 | 2 |
| 5 | 1 | 0 | 0 | 0 | 0 |

**hgp32**

| w | patterns | fail BP | avoid BP | fail BP+OSD | avoid BP+OSD |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 8 | 3 | 3 | 3 | 3 |
| 2 | 28 | 15 | 8 | 15 | 8 |
| 3 | 56 | 31 | 17 | 31 | 17 |
| 4 | 70 | 35 | 25 | 35 | 25 |
| 5 | 56 | 25 | 17 | 25 | 17 |
| 6 | 28 | 13 | 6 | 13 | 6 |
| 7 | 8 | 5 | 3 | 5 | 3 |
| 8 | 1 | 1 | 1 | 1 | 1 |

**hgp33**

| w | patterns | fail BP | avoid BP | fail BP+OSD | avoid BP+OSD |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 13 | 0 | 0 | 0 | 0 |
| 2 | 78 | 27 | 18 | 25 | 16 |
| 3 | 286 | 158 | 67 | 159 | 66 |
| 4 | 715 | 374 | 149 | 383 | 152 |
| 5 | 1287 | 645 | 258 | 645 | 244 |
| 6 | 1716 | 870 | 370 | 858 | 344 |
| 7 | 1716 | 844 | 380 | 838 | 368 |
| 8 | 1287 | 620 | 256 | 622 | 248 |
| 9 | 715 | 354 | 142 | 362 | 140 |
| 10 | 286 | 151 | 68 | 157 | 72 |
| 11 | 78 | 46 | 17 | 43 | 14 |
| 12 | 13 | 6 | 3 | 3 | 0 |
| 13 | 1 | 1 | 0 | 1 | 0 |

## Experiment 2: classification of avoidable failures

### BP+OSD

| code | avoidable | ambiguous | stopping | both | neither |
| --- | --- | --- | --- | --- | --- |
| steane | 0 | 0 | 0 | 0 | 0 |
| rep5 | 0 | 0 | 0 | 0 | 0 |
| hgp22 | 8 | 8 | 2 | 2 | 0 |
| hgp32 | 80 | 80 | 18 | 18 | 0 |
| hgp33 | 1664 | 1664 | 106 | 106 | 0 |

### BP only (convergence split)

| code | avoidable | non-convergence | wrong convergence |
| --- | --- | --- | --- |
| steane | 0 | 0 | 0 |
| rep5 | 0 | 0 | 0 |
| hgp22 | 8 | 8 | 0 |
| hgp32 | 80 | 80 | 0 |
| hgp33 | 1728 | 1216 | 512 |

## Experiment 3: LP calibration as degeneracy detector

| code | solved | frac face | frac vertex | frac ambig | frac unambig | precision (p) | recall (p) | precision (u) | recall (u) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steane | 8 | 3 | 3 | 0 | 3 | 0.000 | n/a | 0.000 | n/a |
| rep5 | 16 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a |
| hgp22 | 4 | 2 | 0 | 2 | 0 | 0.500 | 1.000 | 0.500 | 1.000 |
| hgp32 | 8 | 5 | 0 | 5 | 0 | 0.494 | 1.000 | 0.500 | 1.000 |
| hgp33 | 64 | 33 | 7 | 26 | 7 | 0.403 | 1.000 | 0.394 | 1.000 |

## Findings

- Every avoidable BP+OSD failure on the degenerate codes is a degenerate ambiguity (8/8 on [[5,1,2]] (2 also stopping sets), 80/80 on [[8,1,2]] (18 also stopping sets), 1664/1664 on [[13,1,3]] (106 also stopping sets)); stopping sets add no separate failure mode for BP+OSD, they only co-occur with ambiguity.
- BP-only failures split differently: the distance-2 patches never converge (8 non-convergent and 0 wrongly convergent on [[5,1,2]], 80 non-convergent and 0 wrongly convergent on [[8,1,2]], 1216 non-convergent and 512 wrongly convergent on [[13,1,3]]), and every wrong convergence is a degenerate ambiguity.
- The LP optimal-face signal detects every multi-minimal syndrome (2/2 multi-minimal syndromes on [[5,1,2]] (0 by the vertex signal), 5/5 multi-minimal syndromes on [[8,1,2]] (0 by the vertex signal), 33/33 multi-minimal syndromes on [[13,1,3]] (7 by the vertex signal)); the gap between its precision 47.6% and 100% at p = 0.1 comes from multi-minimal-but-unambiguous syndromes, i.e. degeneracy without ambiguity.
- On the non-degenerate [[7,1,3]] Steane code BP+OSD has no avoidable failures at all, yet 3/8 syndromes show fractional faces: the signal fires but flags nothing.

## Limitations

- Small codes only (n <= 13), code-capacity noise only; the study is a classification exercise, not a decoder benchmark.
- All rates are exact but confined to the X-error side and the five codes listed above.
- The fractional-LP signal is a single HiGHS vertex per syndrome; different optima of the same relaxation may disagree.
- Degenerate-ambiguity labels depend on the logical basis that qudec.logicals returns.
