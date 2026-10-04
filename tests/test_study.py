"""Smoke tests for the study harness and claim generation."""

import unittest

from failuremodes.report import make_claim
from failuremodes.study import StudyConfig, run_code, study_codes


class TestStudySmoke(unittest.TestCase):
    def test_small_codes_consistency(self):
        cfg = StudyConfig(p=0.1, lp=True)
        codes = {n: (a, b) for n, a, b in study_codes()}
        for name in ("steane", "rep5", "hgp22"):
            res = run_code(name, *codes[name], cfg)
            meta = res["meta"]
            self.assertEqual(meta["n_invalid_osd"], 0)
            c = res["classification_bp_osd"]
            self.assertEqual(c["n_ambiguous"] + c["n_stopping"]
                             - c["n_both"] + c["n_neither"], c["n_avoid"])
            cb = res["classification_bp"]
            self.assertEqual(cb["n_non_convergence"]
                             + cb["n_wrong_convergence"], cb["n_avoid"])
            self.assertEqual(
                sum(r["n_patterns"] for r in res["per_weight"]),
                meta["n_patterns"])
            self.assertEqual(
                sum(r["avoid_osd"] for r in res["per_weight"]),
                c["n_avoid"])
            self.assertEqual(res["lp"]["n_solver_failures"], 0)
            self.assertEqual(res["lp"]["n_solved"], res["lp"]["n_syndromes"])

    def test_known_degeneracy_facts(self):
        cfg = StudyConfig(p=0.1, lp=True)
        codes = {n: (a, b) for n, a, b in study_codes()}
        steane = run_code("steane", *codes["steane"], cfg)
        self.assertEqual(steane["meta"]["n_ambiguous_syndromes"], 0)
        self.assertEqual(steane["meta"]["d"], 3)
        rep5 = run_code("rep5", *codes["rep5"], cfg)
        self.assertEqual(rep5["meta"]["n_ambiguous_syndromes"], 0)
        self.assertEqual(rep5["meta"]["d"], 5)
        hgp22 = run_code("hgp22", *codes["hgp22"], cfg)
        self.assertGreaterEqual(hgp22["meta"]["n_ambiguous_syndromes"], 1)
        self.assertEqual(hgp22["meta"]["d"], 2)
        self.assertGreater(hgp22["classification_bp_osd"]["n_avoid"], 0)
        self.assertGreater(hgp22["classification_bp"]["n_avoid"], 0)
        self.assertEqual(hgp22["lp"]["n_fractional"], 2)


class TestClaim(unittest.TestCase):
    def _entry(self, n, k, d, n_avoid, n_ambig, avoid_mass, ambig_mass,
               w_pred, w_both, w_label):
        return {
            "meta": {"n": n, "k": k, "d": d},
            "classification_bp_osd": {
                "n_avoid": n_avoid, "n_ambiguous": n_ambig},
            "weighted": [{
                "p": 0.1, "avoid_osd": avoid_mass,
                "avoid_osd_ambiguous": ambig_mass}],
            "lp": {"p_weighted": {
                "w_pred": w_pred, "w_both": w_both, "w_label": w_label}},
        }

    def test_make_claim_numbers(self):
        results = {
            "config": {"p": 0.1},
            "families": {"degenerate": ["a", "b"]},
            "codes": {
                "a": self._entry(4, 1, 2, 10, 6, 100, 60, 100, 60, 200),
                "b": self._entry(7, 1, 2, 20, 16, 100, 80, 50, 40, 80),
                "steane": self._entry(7, 1, 3, 30, 0, 100, 0, 0, 0, 0),
            },
        }
        claim = make_claim(results)
        self.assertAlmostEqual(claim["ambiguous_share_avoidable_bposd"],
                               0.7)
        self.assertAlmostEqual(claim["lp_precision"], 2 / 3)
        self.assertAlmostEqual(claim["lp_recall"], 100 / 280)
        self.assertIn("70.0%", claim["sentence"])
        self.assertIn("66.7%", claim["sentence"])
        self.assertIn("35.7%", claim["sentence"])
        self.assertIn("[[4,1,2]], [[7,1,2]]", claim["sentence"])
        self.assertIn("0.0%", claim["sentence"])


if __name__ == "__main__":
    unittest.main()
