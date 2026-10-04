"""Brute-force minimal-weight decoding oracle for one side of a CSS code."""

import numpy as np


def enumerate_patterns(n):
    """Return all 2^n binary patterns of length n as an int8 array."""
    idx = np.arange(1 << n, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1
            ).astype(np.int8)


def pack_bits(v):
    """Pack a binary vector to an int with bit i at position i."""
    v = np.asarray(v, dtype=np.int8).reshape(-1)
    powers = 1 << np.arange(len(v), dtype=np.int64)
    return int((v * powers).sum())


def unpack_bits(key, m):
    """Inverse of pack_bits for m-bit vectors."""
    return np.array([(key >> i) & 1 for i in range(m)], dtype=np.int8)


class MldOracle:
    """Exact MLD oracle built by exhaustive enumeration of error patterns.

    For the check matrix h (m, n) and logical matrix l (k, n), enumerate
    all 2^n error patterns once and keep, per syndrome, the minimal-weight
    error patterns and their logical classes. Decoding succeeds for an
    error e when some minimal error for its syndrome is logically
    equivalent to e.
    """

    def __init__(self, h, l):
        h = np.asarray(h, dtype=np.int8)
        l = np.asarray(l, dtype=np.int8)
        m, n = h.shape
        if n > 20:
            raise ValueError(f"exhaustive oracle needs n <= 20, got {n}")
        if n == 0:
            raise ValueError("oracle needs at least one qubit")
        if l.ndim == 1:
            l = l.reshape(1, -1)
        if l.shape[1] != n:
            raise ValueError("logical matrix must have n columns")
        self.h = h
        self.l = l
        self.m = m
        self.n = n
        patterns = enumerate_patterns(n)
        syndromes = (h @ patterns.T) % 2
        weights = patterns.sum(axis=1)
        powers = 1 << np.arange(m, dtype=np.int64)
        keys = (syndromes.T * powers).sum(axis=1)
        order = np.lexsort((weights, keys))
        keys_sorted = keys[order]
        weights_sorted = weights[order]
        patterns_sorted = patterns[order]
        starts = np.concatenate(([True], keys_sorted[1:] != keys_sorted[:-1]))
        group_start = np.nonzero(starts)[0]
        group_end = np.concatenate((group_start[1:], [len(order)]))
        class_powers = 1 << np.arange(l.shape[0], dtype=np.int64)
        self._min_weight = {}
        self._min_errors = {}
        self._classes = {}
        for a, b in zip(group_start, group_end):
            key = int(keys_sorted[a])
            w = int(weights_sorted[a])
            members = patterns_sorted[a:b]
            minimal = members[members.sum(axis=1) == w]
            self._min_weight[key] = w
            self._min_errors[key] = minimal
            packed = ((l @ minimal.T) % 2).T * class_powers
            self._classes[key] = {int(v.sum()) for v in packed}

    def _key(self, s):
        return pack_bits(s)

    def min_weight(self, s):
        """Return the minimal error weight for the syndrome s."""
        return self._min_weight[self._key(s)]

    def min_errors(self, s):
        """Return all minimal-weight error patterns for s as (t, n) int8."""
        return self._min_errors[self._key(s)]

    def n_minimal(self, s):
        """Return the number of minimal errors for s (degeneracy count)."""
        return len(self._min_errors[self._key(s)])

    def n_classes(self, s):
        """Return the number of logical classes among minimal errors of s."""
        return len(self._classes[self._key(s)])

    def is_degenerate(self, s):
        """Return True iff s has more than one minimal error."""
        return self.n_minimal(s) > 1

    def is_ambiguous(self, s):
        """Return True iff minimal errors of s carry several logical classes."""
        return self.n_classes(s) > 1

    def succeeds(self, e):
        """Return True iff some minimal error for syndrome(e) is equivalent to e."""
        e = np.asarray(e, dtype=np.int8).reshape(-1)
        key = self._key((self.h @ e) % 2)
        class_key = pack_bits((self.l @ e) % 2)
        return class_key in self._classes[key]

    def correction(self, e):
        """Return an MLD correction: a minimal error equivalent to e if possible."""
        e = np.asarray(e, dtype=np.int8).reshape(-1)
        key = self._key((self.h @ e) % 2)
        class_key = pack_bits((self.l @ e) % 2)
        for c in self._min_errors[key]:
            if pack_bits((self.l @ c) % 2) == class_key:
                return c
        return self._min_errors[key][0]


def brute_force_distance(h, l):
    """Return the minimum weight of a nontrivial logical operator (one side).

    Enumerate all error patterns; the distance is the minimum weight of a
    pattern with zero syndrome and nontrivial logical action.
    """
    h = np.asarray(h, dtype=np.int8)
    l = np.asarray(l, dtype=np.int8)
    if l.ndim == 1:
        l = l.reshape(1, -1)
    n = h.shape[1]
    if n > 20:
        raise ValueError(f"distance enumeration needs n <= 20, got {n}")
    patterns = enumerate_patterns(n)[1:]
    zero_syn = ~np.any((h @ patterns.T) % 2, axis=0)
    nontrivial = np.any((l @ patterns.T) % 2, axis=0)
    cand = patterns[zero_syn & nontrivial]
    if len(cand) == 0:
        return None
    return int(cand.sum(axis=1).min())
