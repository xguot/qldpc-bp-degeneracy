"""Unit tests for the ADMM study harness and the alpha subclass.

Runs on CPU with a tiny code; the cluster runs the same script with
qudec's decoder on the GPU path.
"""

import unittest

import numpy as np

from qudec.admm import AdmmOsdDecoder
from qudec.codes import steane_code, logicals
from qudec.phenom import PhenomDecoder, benchmark_phenom, sample_phenom

from scripts.run_admm_study import AlphaAdmmOsdDecoder


class TestAdmmHarness(unittest.TestCase):
    """Alpha subclass and phenom wiring on the [[7,1,3]] code."""

    def setUp(self):
        self.h_x, self.h_z = steane_code()
        self.l_x, self.l_z = logicals(self.h_x, self.h_z)
        self.d = 2
        self.shots = 8
        self.seed = 7

    def test_alpha_one_matches_base_decoder(self):
        base = PhenomDecoder(AdmmOsdDecoder, self.h_x, self.h_z, self.l_x,
                             self.l_z, self.d, rho=2.0, osd_order=1)
        sub = PhenomDecoder(AlphaAdmmOsdDecoder, self.h_x, self.h_z,
                            self.l_x, self.l_z, self.d, rho=2.0,
                            osd_order=1, alpha=1.0)
        sx, sz, _, _ = sample_phenom(self.h_x, self.h_z, 0.1, self.d,
                                     self.shots, model="x", seed=self.seed)
        c_base_x, c_base_z = base.decode_corrections(sx, sz)
        c_sub_x, c_sub_z = sub.decode_corrections(sx, sz)
        np.testing.assert_array_equal(c_base_x, c_sub_x)
        np.testing.assert_array_equal(c_base_z, c_sub_z)

    def test_benchmark_phenom_smoke(self):
        decoder = PhenomDecoder(AdmmOsdDecoder, self.h_x, self.h_z,
                                self.l_x, self.l_z, self.d, rho=2.0,
                                osd_order=0)
        res = benchmark_phenom(decoder, self.h_x, self.h_z, self.l_x,
                               self.l_z, 0.05, self.d, self.shots,
                               model="x", seed=self.seed)
        for key in ("ler", "ler_x", "ler_z", "invalid_x", "invalid_z"):
            self.assertIn(key, res)
        self.assertEqual(res["shots"], self.shots)
        # The correction must reproduce the observed detector syndromes.
        self.assertEqual(res["invalid_x"], 0)
        self.assertEqual(res["invalid_z"], 0)

    def test_alpha_rejected_on_ldr_path(self):
        with self.assertRaises(ValueError):
            AlphaAdmmOsdDecoder(self.h_x, self.h_z, self.l_x, self.l_z,
                                ldr=True, alpha=1.5)


if __name__ == "__main__":
    unittest.main()
