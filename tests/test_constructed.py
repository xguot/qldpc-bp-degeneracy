"""Tests for the hypergraph-product constructions."""

import unittest

import numpy as np

from qudec.codes import code_info, logicals

from failuremodes.constructed import hgp_repetition, repetition_classical
from failuremodes.oracle import brute_force_distance


class TestRepetitionClassical(unittest.TestCase):
    def test_shape_and_rank(self):
        h = repetition_classical(5)
        self.assertEqual(h.shape, (4, 5))
        self.assertEqual(np.linalg.matrix_rank(h.astype(int)), 4)


class TestHypergraphProduct(unittest.TestCase):
    def test_shapes_and_css(self):
        cases = {
            (2, 2): (5, 2, 2),
            (3, 2): (8, 4, 3),
            (3, 3): (13, 6, 6),
        }
        for (d1, d2), (n, m_x, m_z) in cases.items():
            h_x, h_z = hgp_repetition(d1, d2)
            self.assertEqual(h_x.shape, (m_x, n))
            self.assertEqual(h_z.shape, (m_z, n))
            self.assertTrue(np.all((h_x @ h_z.T) % 2 == 0))
            self.assertEqual(code_info(h_x, h_z), 1)

    def test_known_distances(self):
        for (d1, d2), d in [((2, 2), 2), ((3, 2), 2), ((3, 3), 3)]:
            h_x, h_z = hgp_repetition(d1, d2)
            _, l_z = logicals(h_x, h_z)
            self.assertEqual(brute_force_distance(h_z, l_z), d)


if __name__ == "__main__":
    unittest.main()
