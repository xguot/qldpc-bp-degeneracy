"""Exact MLD oracle for one CSS side via integer programming (HiGHS MILP).

Replaces exhaustive enumeration for codes with n too large to enumerate.
Minimum-weight decoding of a syndrome is a binary program
min sum(x) s.t. h x = s mod 2, linearized as h x - 2 z = s with integer
z >= 0. Logical-class restrictions add k parity rows with their own
auxiliary variables. Two solves per query give the ambiguity test: the
syndrome is ambiguous iff a minimal error exists in another logical
class.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from failuremodes.oracle import pack_bits


class MldIlpOracle:
    """MLD oracle via integer programming; the interface mirrors MldOracle.

    h: (m, n) check matrix, l: (k, n) logical matrix. Each query solves
    one or two MILPs; results are cached per syndrome and per logical
    class. time_limit bounds every solver call; queries that do not
    prove optimality return "timeout" and count toward n_timeouts.
    """

    def __init__(self, h, l, time_limit=60.0):
        h = np.asarray(h, dtype=np.int8)
        l = np.asarray(l, dtype=np.int8)
        if l.ndim == 1:
            l = l.reshape(1, -1)
        self.h = h
        self.l = l
        self.m, self.n = h.shape
        self.k = l.shape[0]
        self.time_limit = float(time_limit)
        self.n_solves = 0
        self.n_timeouts = 0
        self._cache_w = {}
        self._cache_amb = {}
        self._cache_class = {}
        self._build()

    def _build(self):
        m, n, k = self.m, self.n, self.k
        n_vars = n + m + 2 * k
        rows, cols, data = [], [], []
        for j in range(m):
            supp = np.nonzero(self.h[j])[0]
            for i in supp:
                rows.append(j)
                cols.append(i)
                data.append(1.0)
            rows.append(j)
            cols.append(n + j)
            data.append(-2.0)
        self.a_base = sparse.coo_matrix(
            (data, (rows, cols)), shape=(m, n_vars)).tocsr()
        self.c = np.zeros(n_vars)
        self.c[:n] = 1.0
        self.integrality = np.ones(n_vars)
        z_ub = np.array(
            [np.nonzero(self.h[j])[0].size // 2 for j in range(m)],
            dtype=float)
        y_ub = np.array(
            [np.nonzero(self.l[i])[0].size // 2 + 1 for i in range(k)],
            dtype=float)
        lb = np.zeros(n_vars)
        ub = np.concatenate(
            [np.ones(n), z_ub, y_ub, np.ones(k)])
        self.bounds = Bounds(lb, ub)
        self.l_supp = [np.nonzero(self.l[i])[0] for i in range(k)]

    def _class_rows(self, t, diff):
        """Build the class-constraint rows for target class vector t.

        Equality mode: l x - 2 y_i = t_i. Difference mode adds
        u_i = (l x + t_i) mod 2 and requires sum(u) >= 1.
        """
        m, n, k = self.m, self.n, self.k
        rows, cols, data = [], [], []
        for i in range(k):
            for q in self.l_supp[i]:
                rows.append(i)
                cols.append(q)
                data.append(1.0)
            rows.append(i)
            cols.append(n + m + i)
            data.append(-2.0)
            if diff:
                rows.append(i)
                cols.append(n + m + k + i)
                data.append(-1.0)
        a_class = sparse.coo_matrix(
            (data, (rows, cols)), shape=(k, n + m + 2 * k)).tocsr()
        if not diff:
            return a_class, t.astype(float), t.astype(float)
        u_sum = sparse.coo_matrix(
            (np.ones(k), (np.zeros(k, dtype=int),
                          np.arange(n + m + k, n + m + 2 * k))),
            shape=(1, n + m + 2 * k)).tocsr()
        a = sparse.vstack([a_class, u_sum])
        lb = np.concatenate([(-t).astype(float), [1.0]])
        ub = np.concatenate([(-t).astype(float), [np.inf]])
        return a, lb, ub

    def _solve(self, s, t=None, diff=False):
        """Solve min weight x with h x = s and the class constraint.

        Return (weight, error), None for infeasible, or "timeout" when
        the solver does not prove optimality within the time limit.
        """
        m = self.m
        b = np.asarray(s, dtype=float)
        if t is None:
            a = self.a_base
            lb = b
            ub = b
        else:
            a_class, clb, cub = self._class_rows(t, diff)
            a = sparse.vstack([self.a_base, a_class])
            lb = np.concatenate([b, clb])
            ub = np.concatenate([b, cub])
        constraints = [LinearConstraint(a, lb, ub)]
        self.n_solves += 1
        res = milp(self.c, integrality=self.integrality,
                   bounds=self.bounds, constraints=constraints,
                   options={"time_limit": self.time_limit})
        if res.status == 2:
            return None
        if not res.success or res.x is None:
            self.n_timeouts += 1
            return "timeout"
        e = (np.rint(res.x[:self.n])).astype(np.int8)
        return int(round(res.fun)), e

    def min_weight(self, s):
        """Return (weight, minimal error) for the syndrome, cached."""
        key = pack_bits(s)
        if key not in self._cache_w:
            self._cache_w[key] = self._solve(s)
        return self._cache_w[key]

    def succeeds(self, e):
        """Return True iff some minimal error for syndrome(e) is equivalent
        to e, "timeout" when unresolved, or False."""
        e = np.asarray(e, dtype=np.int8).reshape(-1)
        s = (self.h @ e) % 2
        t = (self.l @ e) % 2
        w = self.min_weight(s)
        if w == "timeout":
            return "timeout"
        cache_key = (pack_bits(s), pack_bits(t))
        if cache_key not in self._cache_class:
            self._cache_class[cache_key] = self._solve(s, t)
        wc = self._cache_class[cache_key]
        if wc == "timeout":
            return "timeout"
        return wc[0] == w[0]

    def is_ambiguous(self, s):
        """Return True iff minimal errors for s span several logical classes,
        "timeout" when unresolved."""
        key = pack_bits(s)
        if key not in self._cache_amb:
            w = self.min_weight(s)
            if w == "timeout" or w is None:
                self._cache_amb[key] = "timeout"
            else:
                t = (self.l @ w[1]) % 2
                w2 = self._solve(s, t, diff=True)
                if w2 == "timeout":
                    self._cache_amb[key] = "timeout"
                elif w2 is None:
                    self._cache_amb[key] = False
                else:
                    self._cache_amb[key] = w2[0] == w[0]
        return self._cache_amb[key]

    def correction(self, e):
        """Return a minimal error equivalent to e, None when unresolved."""
        e = np.asarray(e, dtype=np.int8).reshape(-1)
        s = (self.h @ e) % 2
        t = (self.l @ e) % 2
        cache_key = (pack_bits(s), pack_bits(t))
        if cache_key not in self._cache_class:
            self._cache_class[cache_key] = self._solve(s, t)
        wc = self._cache_class[cache_key]
        if wc == "timeout":
            return None
        return wc[1]
