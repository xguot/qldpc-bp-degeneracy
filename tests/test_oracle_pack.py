"""Regression tests for the int64 bit-packing overflow (m > 63) and the
ILP oracle on the [[144,12,12]] code."""

import unittest

import numpy as np

from qudec.codes import gross_code, logicals
from failuremodes.oracle import pack_bits, unpack_bits
from failuremodes.ilp_oracle import MldIlpOracle


class TestPackBits(unittest.TestCase):
    def test_roundtrip_132_bits(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = (rng.random(132) < 0.5).astype(np.int8)
            np.testing.assert_array_equal(unpack_bits(pack_bits(v), 132), v)

    def test_high_bits_distinct(self):
        # Two syndromes that differ only above bit 63 must not collide.
        a = np.zeros(132, dtype=np.int8)
        b = np.zeros(132, dtype=np.int8)
        b[64] = 1
        self.assertNotEqual(pack_bits(a), pack_bits(b))
        np.testing.assert_array_equal(unpack_bits(pack_bits(b), 132), b)


class TestBb144Oracle(unittest.TestCase):
    """One exact oracle query on bb144: the path that crashed before."""

    def test_min_weight_and_ambiguity(self):
        h_x, h_z = gross_code()
        l_x, l_z = logicals(h_x, h_z)
        oracle = MldIlpOracle(h_z, l_z, time_limit=30.0)
        rng = np.random.default_rng(1)
        e = (rng.random(144) < 0.05).astype(np.int8)
        s = (h_z @ e) % 2
        w = oracle.min_weight(s)
        self.assertNotEqual(w, "timeout")
        self.assertIsNotNone(w)
        self.assertIsInstance(w[0], int)
        self.assertGreaterEqual(w[0], 1)
        a = oracle.is_ambiguous(s)
        self.assertNotEqual(a, "timeout")
        self.assertIn(a, (True, False))


if __name__ == "__main__":
    unittest.main()
