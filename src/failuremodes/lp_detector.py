"""LP-fractionality degeneracy detector built on qudec's exact LP."""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from qudec.lp import lp_formulation, solve_lp


def fractional_stats(x, tol=1e-6):
    """Return (is_fractional, n_fractional, max_violation) for indicators x.

    A coordinate is fractional when it differs from both 0 and 1 by more
    than tol; max_violation is the largest such deviation over all
    coordinates.
    """
    x = np.asarray(x, dtype=float)
    violation = np.minimum(np.abs(x), np.abs(x - 1.0))
    frac = violation > tol
    return bool(frac.any()), int(frac.sum()), float(violation.max())


def lp_solution(h, s):
    """Solve the Gu-Soleimanifar LP for one syndrome via qudec.

    Return a dict with the qubit indicators x, the objective value, and
    the fractional stats, or None when the solver fails. The solution is
    one HiGHS vertex, not a characterization of the whole optimal face.
    """
    try:
        x = solve_lp(h, s)
    except RuntimeError:
        return None
    is_frac, n_frac, max_viol = fractional_stats(x)
    return {
        "x": x,
        "value": float(x.sum()),
        "is_fractional": is_frac,
        "n_fractional": n_frac,
        "max_violation": max_viol,
    }


def lp_face_fractional(h, s, tol=1e-6):
    """Return fractional stats for the whole optimal face of one syndrome.

    Solve the relaxation once for its value v, then bound each qubit
    coordinate on the optimal face {feasible x : sum(x) = v} with 2n
    extra LP solves. The face contains a fractional point iff some
    coordinate's range admits a non-integral value. A syndrome with two
    distinct minimal errors guarantees one: both are optimal vertices
    and their midpoint is fractional, so ambiguous syndromes always
    produce a fractional face. Raise RuntimeError on solver failure.
    """
    h = np.asarray(h, dtype=np.int8)
    s = np.asarray(s, dtype=np.int8)
    x = solve_lp(h, s)
    v = float(np.asarray(x, dtype=float).sum())
    c, a_eq, b_eq = lp_formulation(h, s)
    n = h.shape[1]
    row = sparse.csr_matrix(
        (np.ones(n), (np.zeros(n, dtype=int), np.arange(n))),
        shape=(1, a_eq.shape[1]))
    a_face = sparse.vstack([a_eq, row])
    b_face = np.concatenate([b_eq, [v]])
    lo = np.empty(n)
    hi = np.empty(n)
    for i in range(n):
        ci = np.zeros(a_eq.shape[1])
        ci[i] = 1.0
        res_lo = linprog(ci, A_eq=a_face, b_eq=b_face, bounds=(0, None),
                         method="highs")
        res_hi = linprog(-ci, A_eq=a_face, b_eq=b_face, bounds=(0, None),
                         method="highs")
        if not (res_lo.success and res_hi.success):
            raise RuntimeError("LP face solver failed")
        lo[i] = res_lo.fun
        hi[i] = -res_hi.fun
    cand = np.stack([lo, hi, (lo + hi) / 2], axis=1)
    viol = np.minimum(np.abs(cand), np.abs(cand - 1.0))
    frac = viol.max(axis=1) > tol
    return {
        "value": v,
        "is_fractional": bool(frac.any()),
        "n_fractional": int(frac.sum()),
        "max_violation": float(viol.max()),
        "lo": lo,
        "hi": hi,
        "vertex_fractional": fractional_stats(x)[0],
    }
