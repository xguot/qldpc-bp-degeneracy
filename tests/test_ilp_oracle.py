"""Tests for the integer-programming MLD oracle against brute force."""

import unittest

import numpy as np

from qudec.codes import logicals, repetition_code, steane_code

from failuremodes.constructed import hgp_repetition
from failuremodes.ilp_oracle import MldIlpOracle
from failuremodes.oracle import MldOracle, enumerate_patterns, pack_bits


def assert_oracles_agree(tc, h, l, sample=None):
    """Check the ILP oracle matches the brute-force oracle exactly."""
    brute = MldOracle(h, l)
    ilp = MldIlpOracle(h, l, time_limit=30.0)
    patterns = enumerate_patterns(h.shape[1])
    if sample is not None:
        rng = np.random.default_rng(0)
        patterns = patterns[rng.choice(len(patterns), sample, replace=False)]
    syndromes = (h @ patterns.T) % 2
    for e, s in zip(patterns, syndromes.T):
        tc.assertEqual(ilp.min_weight(s)[0], brute.min_weight(s),
                       f"min weight mismatch for syndrome {pack_bits(s)}")
        tc.assertEqual(ilp.is_ambiguous(s), brute.is_ambiguous(s),
                       f"ambiguity mismatch for syndrome {pack_bits(s)}")
        tc.assertEqual(ilp.succeeds(e), brute.succeeds(e),
                       f"succeeds mismatch for error {e.tolist()}")


class TestIlpOracle(unittest.TestCase):
    def test_steane_matches_brute_force(self):
        h_x, h_z = steane_code()
        _, l_z = logicals(h_x, h_z)
        assert_oracles_agree(self, h_z, l_z)

    def test_rep5_matches_brute_force(self):
        h_x, h_z = repetition_code(5)
        _, l_z = logicals(h_x, h_z)
        assert_oracles_agree(self, h_z, l_z)

    def test_hgp22_matches_brute_force(self):
        h_x, h_z = hgp_repetition(2, 2)
        _, l_z = logicals(h_x, h_z)
        assert_oracles_agree(self, h_z, l_z)

    def test_hgp32_matches_brute_force(self):
        h_x, h_z = hgp_repetition(3, 2)
        _, l_z = logicals(h_x, h_z)
        assert_oracles_agree(self, h_z, l_z)

    def test_hgp33_matches_brute_force_on_sample(self):
        h_x, h_z = hgp_repetition(3, 3)
        _, l_z = logicals(h_x, h_z)
        assert_oracles_agree(self, h_z, l_z, sample=200)


if __name__ == "__main__":
    unittest.main()
