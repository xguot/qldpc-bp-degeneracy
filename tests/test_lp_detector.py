"""Tests for the LP fractional detector."""

import unittest

import numpy as np

from qudec.codes import steane_code

from failuremodes.lp_detector import fractional_stats, lp_face_fractional, lp_solution


class TestFractionalStats(unittest.TestCase):
    def test_integral(self):
        is_frac, n_frac, max_viol = fractional_stats(
            np.array([0.0, 1.0, 0.0]))
        self.assertFalse(is_frac)
        self.assertEqual(n_frac, 0)
        self.assertEqual(max_viol, 0.0)

    def test_fractional(self):
        is_frac, n_frac, max_viol = fractional_stats(
            np.array([0.5, 1.0, 0.0]))
        self.assertTrue(is_frac)
        self.assertEqual(n_frac, 1)
        self.assertAlmostEqual(max_viol, 0.5)

    def test_tolerance(self):
        is_frac, _, _ = fractional_stats(np.array([1e-9]), tol=1e-6)
        self.assertFalse(is_frac)
        is_frac, _, _ = fractional_stats(np.array([1e-9]), tol=1e-12)
        self.assertTrue(is_frac)


class TestLpSolution(unittest.TestCase):
    def setUp(self):
        self.h = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)

    def test_value_matches_minimal_weight(self):
        for s, expect in [([1, 0], 1.0), ([0, 1], 1.0),
                          ([1, 1], 2.0), ([0, 0], 0.0)]:
            res = lp_solution(self.h, np.array(s, dtype=np.int8))
            self.assertIsNotNone(res)
            self.assertAlmostEqual(res["value"], expect, places=6)
            self.assertTrue(np.all(res["x"] >= -1e-9))
            self.assertTrue(np.all(res["x"] <= 1 + 1e-9))

    def test_zero_syndrome_integral(self):
        res = lp_solution(self.h, np.zeros(2, dtype=np.int8))
        self.assertFalse(res["is_fractional"])
        self.assertAlmostEqual(res["value"], 0.0, places=6)

    def test_steane_weight_one_value(self):
        h_x, h_z = steane_code()
        e = np.zeros(7, dtype=np.int8)
        e[0] = 1
        res = lp_solution(h_z, (h_z @ e) % 2)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["value"], 1.0, places=6)


class TestLpFace(unittest.TestCase):
    def setUp(self):
        self.h = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)

    def test_ambiguous_syndrome_face_is_fractional(self):
        face = lp_face_fractional(self.h, np.array([1, 0], dtype=np.int8))
        self.assertTrue(face["is_fractional"])
        self.assertAlmostEqual(face["value"], 1.0, places=6)

    def test_zero_syndrome_face_is_integral(self):
        face = lp_face_fractional(self.h, np.zeros(2, dtype=np.int8))
        self.assertFalse(face["is_fractional"])
        self.assertAlmostEqual(face["value"], 0.0, places=6)

    def test_two_minimal_errors_guarantee_fractional_face(self):
        h = np.array([[1, 1, 0, 0, 1], [0, 0, 1, 1, 1]], dtype=np.int8)
        s = np.array([1, 0], dtype=np.int8)
        face = lp_face_fractional(h, s)
        self.assertTrue(face["is_fractional"])
        self.assertAlmostEqual(face["value"], 1.0, places=6)
        self.assertAlmostEqual(face["lo"][0], 0.0, places=6)
        self.assertAlmostEqual(face["hi"][0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
