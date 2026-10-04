"""Tests for the brute-force MLD oracle."""

import unittest

import numpy as np

from qudec.codes import logicals, repetition_code, steane_code

from failuremodes.oracle import (
    MldOracle,
    brute_force_distance,
    enumerate_patterns,
    pack_bits,
    unpack_bits,
)


class TestPackBits(unittest.TestCase):
    def test_roundtrip(self):
        for m in (1, 3, 8):
            for k in range(1 << m):
                self.assertEqual(pack_bits(unpack_bits(k, m)), k)


class TestMldOracleHandBuilt(unittest.TestCase):
    def setUp(self):
        self.h = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)
        self.l = np.array([[1, 0, 0, 0]], dtype=np.int8)
        self.oracle = MldOracle(self.h, self.l)

    def test_ambiguous_syndrome(self):
        s10 = np.array([1, 0], dtype=np.int8)
        self.assertEqual(self.oracle.min_weight(s10), 1)
        self.assertEqual(self.oracle.n_minimal(s10), 2)
        self.assertEqual(self.oracle.n_classes(s10), 2)
        self.assertTrue(self.oracle.is_degenerate(s10))
        self.assertTrue(self.oracle.is_ambiguous(s10))

    def test_cross_syndrome(self):
        s11 = np.array([1, 1], dtype=np.int8)
        self.assertEqual(self.oracle.min_weight(s11), 2)
        self.assertEqual(self.oracle.n_minimal(s11), 4)
        self.assertTrue(self.oracle.is_ambiguous(s11))

    def test_zero_syndrome(self):
        s00 = np.array([0, 0], dtype=np.int8)
        self.assertEqual(self.oracle.min_weight(s00), 0)
        self.assertFalse(self.oracle.is_degenerate(s00))
        self.assertFalse(self.oracle.is_ambiguous(s00))

    def test_succeeds_semantics(self):
        e0 = np.array([1, 0, 0, 0], dtype=np.int8)
        e1 = np.array([0, 1, 0, 0], dtype=np.int8)
        logical = e0 ^ e1
        self.assertTrue(self.oracle.succeeds(e0))
        self.assertTrue(self.oracle.succeeds(e1))
        self.assertFalse(self.oracle.succeeds(logical))

    def test_correction_picks_equivalent_minimal(self):
        e1 = np.array([0, 1, 0, 0], dtype=np.int8)
        corr = self.oracle.correction(e1)
        self.assertTrue(np.array_equal(corr, e1))


class TestMldOracleCodes(unittest.TestCase):
    def test_steane_weight_one_unique_minimal(self):
        h_x, h_z = steane_code()
        _, l_z = logicals(h_x, h_z)
        oracle = MldOracle(h_z, l_z)
        for i in range(7):
            e = np.zeros(7, dtype=np.int8)
            e[i] = 1
            s = (h_z @ e) % 2
            self.assertEqual(oracle.n_minimal(s), 1)
            self.assertEqual(oracle.min_weight(s), 1)
            self.assertFalse(oracle.is_ambiguous(s))

    def test_steane_no_ambiguous_syndromes(self):
        h_x, h_z = steane_code()
        _, l_z = logicals(h_x, h_z)
        oracle = MldOracle(h_z, l_z)
        syndromes = (h_z @ enumerate_patterns(7).T) % 2
        for s in syndromes.T:
            self.assertFalse(oracle.is_ambiguous(s))

    def test_rep5_unique_minimal_per_syndrome(self):
        """The rep code has distinct weight-1 syndromes and its coset
        pairs (e, e xor X_L) differ in weight by 5, so every syndrome has
        exactly one minimal error and no ambiguous syndrome."""
        h_x, h_z = repetition_code(5)
        _, l_z = logicals(h_x, h_z)
        oracle = MldOracle(h_z, l_z)
        syndromes = (h_z @ enumerate_patterns(5).T) % 2
        for s in syndromes.T:
            self.assertEqual(oracle.n_minimal(s), 1)
            self.assertFalse(oracle.is_ambiguous(s))
            self.assertFalse(oracle.is_degenerate(s))
        for i in range(5):
            e = np.zeros(5, dtype=np.int8)
            e[i] = 1
            self.assertEqual(oracle.min_weight((h_z @ e) % 2), 1)

    def test_brute_force_distance_steane(self):
        h_x, h_z = steane_code()
        _, l_z = logicals(h_x, h_z)
        self.assertEqual(brute_force_distance(h_z, l_z), 3)


if __name__ == "__main__":
    unittest.main()
