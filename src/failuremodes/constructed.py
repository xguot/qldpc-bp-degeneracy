"""Hypergraph-product CSS code constructions for the failure-mode study."""

import numpy as np


def hypergraph_product(h1, h2):
    """Return (h_x, h_z) for the hypergraph product of two classical codes.

    h1: (m1, n1) parity-check matrix, h2: (m2, n2). The quantum code has
    n = n1 n2 + m1 m2 qubits and k = k1 k2 + k1^T k2^T logical qubits:

        h_x = [h1 (x) I_n2 | I_m1 (x) h2^T]
        h_z = [I_n1 (x) h2  | h1^T (x) I_m2]

    The CSS condition h_x h_z^T = 0 holds by construction. For two
    repetition codes this gives the unrotated surface-code patch
    [[d1 d2 + (d1 - 1) (d2 - 1), 1, min(d1, d2)]].
    """
    m1, n1 = h1.shape
    m2, n2 = h2.shape
    h_x = np.concatenate(
        [np.kron(h1, np.eye(n2, dtype=np.int8)),
         np.kron(np.eye(m1, dtype=np.int8), h2.T)], axis=1)
    h_z = np.concatenate(
        [np.kron(np.eye(n1, dtype=np.int8), h2),
         np.kron(h1.T, np.eye(m2, dtype=np.int8))], axis=1)
    return h_x, h_z


def repetition_classical(d):
    """Return the (d - 1, d) parity-check matrix of the [d, 1, d] code."""
    h = np.zeros((d - 1, d), dtype=np.int8)
    for i in range(d - 1):
        h[i, i] = 1
        h[i, i + 1] = 1
    return h


def hgp_repetition(d1, d2):
    """Return (h_x, h_z) for the hypergraph product of two repetition codes.

    Gives a surface-code patch with n = d1 d2 + (d1 - 1) (d2 - 1)
    qubits, one logical qubit, and distance min(d1, d2).
    """
    return hypergraph_product(repetition_classical(d1),
                              repetition_classical(d2))
