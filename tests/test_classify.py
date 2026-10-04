"""Tests for the failure classifiers on hand-constructed cases."""

import unittest

import numpy as np

from failuremodes.classify import (
    check_satisfied,
    convergence_label,
    is_stopping_set,
    logical_success,
)


class TestStoppingSet(unittest.TestCase):
    def setUp(self):
        self.h = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)

    def test_each_check_touches_two(self):
        self.assertTrue(is_stopping_set(self.h, [1, 1, 1, 1]))

    def test_single_pair_is_stopping(self):
        self.assertTrue(is_stopping_set(self.h, [1, 1, 0, 0]))

    def test_cross_pattern_touches_one_each(self):
        self.assertFalse(is_stopping_set(self.h, [1, 0, 1, 0]))

    def test_empty_pattern_vacuous(self):
        self.assertTrue(is_stopping_set(self.h, [0, 0, 0, 0]))

    def test_rep_code_single_flip_not_stopping(self):
        h = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=np.int8)
        self.assertFalse(is_stopping_set(h, [1, 0, 0, 0, 0]))
        self.assertTrue(is_stopping_set(h, [1, 1, 1, 1, 1]))


class TestConvergence(unittest.TestCase):
    def setUp(self):
        self.h = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8)

    def test_check_satisfied(self):
        s = np.array([1, 0], dtype=np.int8)
        self.assertTrue(check_satisfied(self.h, s, [1, 0, 0, 0]))
        self.assertTrue(check_satisfied(self.h, s, [0, 1, 0, 0]))
        self.assertFalse(check_satisfied(self.h, s, [0, 0, 0, 0]))
        self.assertFalse(check_satisfied(self.h, s, [1, 0, 1, 0]))

    def test_convergence_label(self):
        s = np.array([1, 0], dtype=np.int8)
        self.assertEqual(convergence_label(self.h, s, [1, 0, 0, 0]),
                         "wrong_convergence")
        self.assertEqual(convergence_label(self.h, s, [0, 0, 0, 0]),
                         "non_convergence")


class TestLogicalSuccess(unittest.TestCase):
    def test_equivalent_and_inequivalent(self):
        l = np.array([[1, 0, 0, 0]], dtype=np.int8)
        self.assertTrue(logical_success(l, [0, 1, 0, 0], [0, 0, 0, 0]))
        self.assertTrue(logical_success(l, [1, 1, 0, 0], [1, 0, 0, 0]))
        self.assertFalse(logical_success(l, [1, 0, 0, 0], [0, 1, 0, 0]))


if __name__ == "__main__":
    unittest.main()
