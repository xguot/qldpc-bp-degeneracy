"""Failure classifiers for decoded error instances."""

import numpy as np


def is_stopping_set(h, pattern):
    """Return True iff the support of pattern is a stopping set of h.

    Every check adjacent to at least one flipped position must touch at
    least two flipped positions. The empty pattern satisfies the
    condition vacuously. For a syndrome-valid residual this test is
    vacuous (every codeword support is a stopping set), so the study
    applies it to the true error support, matching the classical BEC
    stopping-set analysis.
    """
    h = np.asarray(h, dtype=np.int8)
    pattern = np.asarray(pattern, dtype=np.int8).reshape(-1)
    support = np.nonzero(pattern)[0]
    if len(support) == 0:
        return True
    counts = h[:, support].sum(axis=1)
    return bool(np.all(counts != 1))


def check_satisfied(h, s, correction):
    """Return True iff the correction reproduces the syndrome on h."""
    h = np.asarray(h, dtype=np.int8)
    s = np.asarray(s, dtype=np.int8).reshape(-1)
    correction = np.asarray(correction, dtype=np.int8).reshape(-1)
    return bool(np.array_equal((h @ correction) % 2, s % 2))


def logical_success(l, error, correction):
    """Return True iff error and correction differ only by a stabilizer."""
    l = np.asarray(l, dtype=np.int8)
    if l.ndim == 1:
        l = l.reshape(1, -1)
    resid = (np.asarray(error, dtype=np.int8) ^
             np.asarray(correction, dtype=np.int8)) % 2
    return bool(~np.any((l @ resid) % 2))


def convergence_label(h, s, correction):
    """Classify a failed decode as wrong convergence or non-convergence.

    Precondition: the correction failed logically. A correction that
    satisfies all checks is a wrong convergence; one that does not means
    the decoder stopped unsatisfied (non-convergence).
    """
    if check_satisfied(h, s, correction):
        return "wrong_convergence"
    return "non_convergence"
