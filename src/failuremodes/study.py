"""Failure-mode study harness: experiments 1-3 of failuremodes_plan.md.

All study codes have n <= 12, so every experiment runs exhaustively over
all 2^n error patterns on the X-error side; there is no sampling and the
whole study is deterministic. BP and BP+OSD results are reported
separately throughout.
"""

from dataclasses import asdict, dataclass

import numpy as np

from qudec.bposd import BpOsdDecoder
from qudec.codes import code_info, logicals, repetition_code, steane_code

from failuremodes.bp_only import BpOnlyDecoder
from failuremodes.classify import check_satisfied
from failuremodes.constructed import hgp_repetition
from failuremodes.lp_detector import lp_face_fractional
from failuremodes.oracle import (
    MldOracle,
    brute_force_distance,
    enumerate_patterns,
    unpack_bits,
)
from failuremodes.report import make_claim


@dataclass
class StudyConfig:
    """Configuration of one study run.

    p: channel error rate feeding the decoder LLRs and the p-weighted
    aggregates. p_values: rates for the weighted aggregate table.
    lp_max_syndromes: cap on distinct syndromes sent to the LP solver;
    all study codes stay below it, so it only guards future codes.
    """

    p: float = 0.1
    p_values: tuple = (0.05, 0.1)
    osd_order: int = 1
    bp_max_iter: int = 30
    lp: bool = True
    lp_max_syndromes: int = 4096
    seed: int = 0


def study_codes():
    """Return (name, h_x, h_z) for every code of the study."""
    return [
        ("steane", *steane_code()),
        ("rep5", *repetition_code(5)),
        ("hgp22", *hgp_repetition(2, 2)),
        ("hgp32", *hgp_repetition(3, 2)),
        ("hgp33", *hgp_repetition(3, 3)),
    ]


def _logical_ok(l, patterns, corrections):
    resid = (patterns ^ corrections) % 2
    return ~np.any((l @ resid.T) % 2, axis=0)


def _syndrome_keys(syndromes):
    powers = 1 << np.arange(syndromes.shape[0], dtype=np.int64)
    return (syndromes.T * powers).sum(axis=1)


def _precision_recall(pred, label, w):
    """Return precision/recall with counts and masses; w None means uniform."""
    pred = np.asarray(pred, dtype=bool)
    label = np.asarray(label, dtype=bool)
    w = np.ones(len(pred)) if w is None else np.asarray(w, dtype=float)
    w_pred = float(w[pred].sum())
    w_label = float(w[label].sum())
    w_both = float(w[pred & label].sum())
    return {
        "precision": None if w_pred == 0 else w_both / w_pred,
        "recall": None if w_label == 0 else w_both / w_label,
        "n_pred": int(pred.sum()),
        "n_label": int(label.sum()),
        "n_both": int((pred & label).sum()),
        "w_pred": w_pred,
        "w_label": w_label,
        "w_both": w_both,
    }


def run_code(name, h_x, h_z, cfg):
    """Run experiments 1-3 exhaustively on the X-error side of one code."""
    l_x, l_z = logicals(h_x, h_z)
    n = h_x.shape[1]
    h, l = h_z, l_z
    oracle = MldOracle(h, l)
    dec_bp = BpOnlyDecoder(h_x, h_z, l_x, l_z, cfg.p, cfg.p,
                           max_iter=cfg.bp_max_iter)
    dec_osd = BpOsdDecoder(h_x, h_z, l_x, l_z, cfg.p, cfg.p,
                           max_iter=cfg.bp_max_iter, osd_order=cfg.osd_order)
    patterns = enumerate_patterns(n)
    weights = patterns.sum(axis=1)
    syndromes = (h @ patterns.T) % 2
    empty_z = np.zeros((len(patterns), h_x.shape[0]), dtype=np.int8)
    c_bp, _ = dec_bp.decode_corrections(
        syndromes.T.astype(np.int8), empty_z)
    c_osd, _ = dec_osd.decode_corrections(
        syndromes.T.astype(np.int8), empty_z)
    ok_bp = _logical_ok(l, patterns, c_bp)
    ok_osd = _logical_ok(l, patterns, c_osd)
    valid_bp = np.all((h @ c_bp.T) % 2 == syndromes, axis=0)
    valid_osd = np.all((h @ c_osd.T) % 2 == syndromes, axis=0)
    oracle_ok = np.array([oracle.succeeds(p) for p in patterns], dtype=bool)
    ambiguous = np.array([oracle.is_ambiguous(s) for s in syndromes.T],
                         dtype=bool)
    n_minimal = np.array([oracle.n_minimal(s) for s in syndromes.T],
                         dtype=np.int64)
    overlaps = h.astype(np.int32) @ patterns.T.astype(np.int32)
    stopping = ~np.any(overlaps == 1, axis=0)
    fail_bp = ~ok_bp
    fail_osd = ~ok_osd
    avoid_bp = oracle_ok & fail_bp
    avoid_osd = oracle_ok & fail_osd

    per_weight = []
    for w in range(n + 1):
        sel = weights == w
        per_weight.append({
            "w": w,
            "n_patterns": int(sel.sum()),
            "fail_bp": int((fail_bp & sel).sum()),
            "avoid_bp": int((avoid_bp & sel).sum()),
            "fail_osd": int((fail_osd & sel).sum()),
            "avoid_osd": int((avoid_osd & sel).sum()),
        })

    weighted = []
    for p in cfg.p_values:
        prob = (p ** weights) * ((1 - p) ** (n - weights))
        weighted.append({
            "p": p,
            "fail_bp": float((prob * fail_bp).sum()),
            "avoid_bp": float((prob * avoid_bp).sum()),
            "fail_osd": float((prob * fail_osd).sum()),
            "avoid_osd": float((prob * avoid_osd).sum()),
            "avoid_osd_ambiguous": float(
                (prob * (avoid_osd & ambiguous)).sum()),
            "avoid_osd_stopping": float(
                (prob * (avoid_osd & stopping)).sum()),
        })

    cls_osd = {
        "n_avoid": int(avoid_osd.sum()),
        "n_ambiguous": int((avoid_osd & ambiguous).sum()),
        "n_stopping": int((avoid_osd & stopping).sum()),
        "n_both": int((avoid_osd & ambiguous & stopping).sum()),
        "n_neither": int((avoid_osd & ~ambiguous & ~stopping).sum()),
    }
    cls_bp = {
        "n_avoid": int(avoid_bp.sum()),
        "n_non_convergence": int((avoid_bp & ~valid_bp).sum()),
        "n_wrong_convergence": int((avoid_bp & valid_bp).sum()),
        "wrong_convergence_ambiguous": int(
            (avoid_bp & valid_bp & ambiguous).sum()),
        "wrong_convergence_stopping": int(
            (avoid_bp & valid_bp & stopping).sum()),
    }

    unique_keys = sorted({int(k) for k in _syndrome_keys(syndromes)})
    n_degenerate_syn = 0
    n_ambiguous_syn = 0
    max_multiplicity = 0
    for k in unique_keys:
        s = unpack_bits(k, h.shape[0])
        if oracle.n_minimal(s) > 1:
            n_degenerate_syn += 1
        if oracle.is_ambiguous(s):
            n_ambiguous_syn += 1
        max_multiplicity = max(max_multiplicity, oracle.n_minimal(s))

    lp = None
    if cfg.lp:
        keys = unique_keys
        if len(keys) > cfg.lp_max_syndromes:
            rng = np.random.default_rng(cfg.seed)
            keys = sorted(rng.choice(keys, cfg.lp_max_syndromes,
                                     replace=False).tolist())
        frac_by_key = {}
        n_solver_fail = 0
        n_frac_vertex = 0
        for k in keys:
            s = unpack_bits(k, h.shape[0])
            try:
                face = lp_face_fractional(h, s)
            except RuntimeError:
                n_solver_fail += 1
                frac_by_key[k] = None
                continue
            frac_by_key[k] = face["is_fractional"]
            n_frac_vertex += int(face["vertex_fractional"])
        frac_inst = np.array([frac_by_key.get(int(k)) for k in
                              _syndrome_keys(syndromes)])
        known = np.array([v is not None for v in frac_inst])
        prob = (cfg.p ** weights) * ((1 - cfg.p) ** (n - weights))
        lp = {
            "n_syndromes": len(keys),
            "n_solved": len(keys) - n_solver_fail,
            "n_solver_failures": n_solver_fail,
            "n_fractional": int(sum(1 for v in frac_by_key.values() if v)),
            "n_fractional_vertex": n_frac_vertex,
            "uniform": _precision_recall(frac_inst[known],
                                         avoid_osd[known], None),
            "p_weighted": _precision_recall(frac_inst[known],
                                            avoid_osd[known], prob[known]),
        }
        n_frac_ambig = 0
        n_frac_unambig = 0
        for k in keys:
            if frac_by_key.get(k) is None:
                continue
            if frac_by_key[k]:
                if oracle.is_ambiguous(unpack_bits(k, h.shape[0])):
                    n_frac_ambig += 1
                else:
                    n_frac_unambig += 1
        lp["n_frac_ambiguous_syn"] = n_frac_ambig
        lp["n_frac_unambiguous_syn"] = n_frac_unambig

    meta = {
        "name": name,
        "n": n,
        "m_x": int(h_x.shape[0]),
        "m_z": int(h.shape[0]),
        "k": int(code_info(h_x, h_z)),
        "d": brute_force_distance(h, l),
        "n_patterns": int(len(patterns)),
        "n_unique_syndromes": len(unique_keys),
        "n_degenerate_syndromes": n_degenerate_syn,
        "n_ambiguous_syndromes": n_ambiguous_syn,
        "max_minimal_multiplicity": max_multiplicity,
        "n_invalid_bp": int((~valid_bp).sum()),
        "n_invalid_osd": int((~valid_osd).sum()),
        "n_minimal_gt1_instances": int((n_minimal > 1).sum()),
    }
    return {
        "meta": meta,
        "per_weight": per_weight,
        "weighted": weighted,
        "classification_bp_osd": cls_osd,
        "classification_bp": cls_bp,
        "lp": lp,
    }


def run_study(cfg=None):
    """Run the whole study and attach the claim sentence."""
    cfg = cfg or StudyConfig()
    codes = {}
    for name, h_x, h_z in study_codes():
        codes[name] = run_code(name, h_x, h_z, cfg)
    results = {
        "config": asdict(cfg),
        "codes": codes,
        "families": {
            "degenerate": ["hgp22", "hgp32", "hgp33"],
            "controls": ["steane", "rep5"],
        },
    }
    results["claim"] = make_claim(results)
    return results
