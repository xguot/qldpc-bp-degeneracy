"""Failure-mode study harness: experiments 1-3 of failuremodes_plan.md.

Two modes per code: exhaustive enumeration over all 2^n patterns
(n <= 13 study codes, deterministic) and i.i.d. sampling with fixed
seeds for larger codes (HGP surface patches up to n = 85 and the
bivariate bicycle codes), where the MLD oracle is an integer program.
BP and BP+OSD results are reported separately throughout.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from qudec.bposd import BpOsdDecoder
from qudec.codes import (
    code_info,
    gross_code,
    logicals,
    medium_code,
    repetition_code,
    steane_code,
)
from qudec.noise import sample_iid_errors

from failuremodes.bp_only import BpOnlyDecoder
from failuremodes.classify import check_satisfied
from failuremodes.constructed import hgp_repetition
from failuremodes.ilp_oracle import MldIlpOracle
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

    p: channel error rate feeding the decoder LLRs and the aggregates.
    p_values: rates for the weighted aggregate table (exhaustive mode).
    codes: registry names to run; shots > 0 selects the sampled path.
    oracle: "ilp" or "none" for the MLD oracle in sampled mode.
    ilp_time_limit: per-solve wall-clock cap for the MILP oracle.
    lp_max_syndromes: cap on distinct syndromes sent to the LP solver.
    """

    p: float = 0.1
    p_values: tuple = (0.05, 0.1)
    codes: tuple = ("steane", "rep5", "hgp22", "hgp32", "hgp33")
    shots: int = 0
    oracle: str = "ilp"
    ilp_time_limit: float = 60.0
    lp_max_syndromes: int = 4096
    osd_order: int = 1
    bp_max_iter: int = 30
    lp: bool = True
    seed: int = 0
    device: str = "auto"
    gpu_chunk: int = 10000


def code_registry():
    """Return name -> (builder, analytic distance) for every study code."""
    return {
        "steane": (steane_code, 3),
        "rep5": (lambda: repetition_code(5), 5),
        "hgp22": (lambda: hgp_repetition(2, 2), 2),
        "hgp32": (lambda: hgp_repetition(3, 2), 2),
        "hgp33": (lambda: hgp_repetition(3, 3), 3),
        "hgp44": (lambda: hgp_repetition(4, 4), 4),
        "hgp55": (lambda: hgp_repetition(5, 5), 5),
        "hgp66": (lambda: hgp_repetition(6, 6), 6),
        "hgp77": (lambda: hgp_repetition(7, 7), 7),
        "bb72": (medium_code, 6),
        "bb144": (gross_code, 12),
    }


def study_codes():
    """Return (name, h_x, h_z) for the default exhaustive study codes."""
    registry = code_registry()
    names = ("steane", "rep5", "hgp22", "hgp32", "hgp33")
    return [(n, *registry[n][0]()) for n in names]


def _logical_ok(l, patterns, corrections):
    resid = (patterns ^ corrections) % 2
    return ~np.any((l @ resid.T) % 2, axis=0)


def _syndrome_keys(syndromes):
    # object dtype: m > 63 overflows int64 shifts
    powers = 1 << np.arange(syndromes.shape[0], dtype=object)
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


def _lp_calibration(h, syndromes, labels, weights, cfg):
    """Run the LP face calibration on capped distinct sampled syndromes."""
    unique_keys = sorted({int(k) for k in _syndrome_keys(syndromes)})
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
    uniform = _precision_recall(frac_inst[known], labels[known], None)
    return {
        "n_syndromes": len(keys),
        "n_solved": len(keys) - n_solver_fail,
        "n_solver_failures": n_solver_fail,
        "n_fractional": int(sum(1 for v in frac_by_key.values() if v)),
        "n_fractional_vertex": n_frac_vertex,
        "uniform": uniform,
        "p_weighted": _precision_recall(frac_inst[known], labels[known],
                                        weights[known]),
    }, frac_by_key


def _sampled_decoders(h_x, h_z, l_x, l_z, cfg):
    """Return (bp_decoder, osd_decoder, device) for the sampled path."""
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but torch reports no CUDA")
    use_gpu = cfg.device == "cuda" or (
        cfg.device == "auto" and torch.cuda.is_available())
    if use_gpu:
        from failuremodes.gpu_decoders import (
            GpuBpOnlyDecoder,
            GpuBpOsdDecoder,
        )
        bp = GpuBpOnlyDecoder(h_x, h_z, l_x, l_z, cfg.p, cfg.p,
                              max_iter=cfg.bp_max_iter, device="cuda",
                              gpu_chunk=cfg.gpu_chunk)
        osd = GpuBpOsdDecoder(h_x, h_z, l_x, l_z, cfg.p, cfg.p,
                              max_iter=cfg.bp_max_iter,
                              osd_order=cfg.osd_order, device="cuda",
                              gpu_chunk=cfg.gpu_chunk)
        return bp, osd, "cuda"
    bp = BpOnlyDecoder(h_x, h_z, l_x, l_z, cfg.p, cfg.p,
                       max_iter=cfg.bp_max_iter)
    osd = BpOsdDecoder(h_x, h_z, l_x, l_z, cfg.p, cfg.p,
                       max_iter=cfg.bp_max_iter, osd_order=cfg.osd_order)
    return bp, osd, "cpu"


def run_code(name, h_x, h_z, cfg, d_analytic=None):
    """Run experiments 1-3 on the X-error side of one code.

    Exhaustive over all 2^n patterns when cfg.shots == 0, sampled
    otherwise.
    """
    if cfg.shots > 0:
        return run_code_sampled(name, h_x, h_z, cfg, d_analytic)
    return run_code_exhaustive(name, h_x, h_z, cfg)


def run_code_exhaustive(name, h_x, h_z, cfg):
    """Exhaustive path: all 2^n patterns, brute-force oracle."""
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
        prob = (cfg.p ** weights) * ((1 - cfg.p) ** (n - weights))
        lp, frac_by_key = _lp_calibration(h, syndromes, avoid_osd, prob, cfg)
        n_frac_ambig = 0
        n_frac_unambig = 0
        for k, v in frac_by_key.items():
            if v is None or not v:
                continue
            if oracle.is_ambiguous(unpack_bits(k, h.shape[0])):
                n_frac_ambig += 1
            else:
                n_frac_unambig += 1
        lp["n_frac_ambiguous_syn"] = n_frac_ambig
        lp["n_frac_unambiguous_syn"] = n_frac_unambig
        lp["n_ambiguous_syn"] = n_ambiguous_syn
        lp["n_unambiguous_syn"] = len(unique_keys) - n_ambiguous_syn
        lp["label"] = "avoidable"

    meta = {
        "name": name,
        "sampled": False,
        "n": n,
        "m_x": int(h_x.shape[0]),
        "m_z": int(h.shape[0]),
        "k": int(code_info(h_x, h_z)),
        "d": brute_force_distance(h, l),
        "n_patterns": int(len(patterns)),
        "shots": int(len(patterns)),
        "n_unique_syndromes": len(unique_keys),
        "n_degenerate_syndromes": n_degenerate_syn,
        "n_ambiguous_syndromes": n_ambiguous_syn,
        "max_minimal_multiplicity": max_multiplicity,
        "n_invalid_bp": int((~valid_bp).sum()),
        "n_invalid_osd": int((~valid_osd).sum()),
        "n_minimal_gt1_instances": int((n_minimal > 1).sum()),
        "p": cfg.p,
        "seed": cfg.seed,
    }
    return {
        "meta": meta,
        "per_weight": per_weight,
        "weighted": weighted,
        "classification_bp_osd": cls_osd,
        "classification_bp": cls_bp,
        "lp": lp,
    }


def run_code_sampled(name, h_x, h_z, cfg, d_analytic=None):
    """Sampled path: i.i.d. shots with a fixed seed, ILP MLD oracle."""
    l_x, l_z = logicals(h_x, h_z)
    n = h_x.shape[1]
    h, l = h_z, l_z
    use_oracle = cfg.oracle != "none"
    oracle = (MldIlpOracle(h, l, time_limit=cfg.ilp_time_limit)
              if use_oracle else None)
    dec_bp, dec_osd, device = _sampled_decoders(h_x, h_z, l_x, l_z, cfg)
    patterns, _ = sample_iid_errors(n, cfg.p, cfg.shots, "x", seed=cfg.seed)
    syndromes = (h @ patterns.T) % 2
    empty_z = np.zeros((cfg.shots, h_x.shape[0]), dtype=np.int8)
    c_bp, _ = dec_bp.decode_corrections(
        syndromes.T.astype(np.int8), empty_z)
    c_osd, _ = dec_osd.decode_corrections(
        syndromes.T.astype(np.int8), empty_z)
    ok_bp = _logical_ok(l, patterns, c_bp)
    ok_osd = _logical_ok(l, patterns, c_osd)
    valid_bp = np.all((h @ c_bp.T) % 2 == syndromes, axis=0)
    valid_osd = np.all((h @ c_osd.T) % 2 == syndromes, axis=0)
    overlaps = h.astype(np.int32) @ patterns.T.astype(np.int32)
    stopping = ~np.any(overlaps == 1, axis=0)
    fail_bp = ~ok_bp
    fail_osd = ~ok_osd

    if use_oracle:
        need = fail_bp | fail_osd
        oracle_ok = np.zeros(cfg.shots, dtype=bool)
        ambiguous = np.zeros(cfg.shots, dtype=bool)
        resolved = np.ones(cfg.shots, dtype=bool)
        for i in np.nonzero(need)[0]:
            s = syndromes[:, i]
            o = oracle.succeeds(patterns[i])
            a = oracle.is_ambiguous(s)
            if o == "timeout" or a == "timeout":
                resolved[i] = False
                continue
            oracle_ok[i] = bool(o)
            ambiguous[i] = bool(a)
        avoid_bp = fail_bp & resolved & oracle_ok
        avoid_osd = fail_osd & resolved & oracle_ok
    else:
        avoid_bp = None
        avoid_osd = None
        ambiguous = np.zeros(cfg.shots, dtype=bool)

    shots = float(cfg.shots)
    weighted = [{
        "p": cfg.p,
        "fail_bp": float(fail_bp.mean()),
        "avoid_bp": (None if avoid_bp is None else float(avoid_bp.mean())),
        "fail_osd": float(fail_osd.mean()),
        "avoid_osd": (None if avoid_osd is None else float(avoid_osd.mean())),
        "avoid_osd_ambiguous": (
            None if avoid_osd is None
            else float((avoid_osd & ambiguous).mean())),
        "avoid_osd_stopping": (
            None if avoid_osd is None
            else float((avoid_osd & stopping).mean())),
        "shots": cfg.shots,
    }]

    if use_oracle:
        cls_osd = {
            "n_avoid": int(avoid_osd.sum()),
            "n_ambiguous": int((avoid_osd & ambiguous).sum()),
            "n_stopping": int((avoid_osd & stopping).sum()),
            "n_both": int((avoid_osd & ambiguous & stopping).sum()),
            "n_neither": int((avoid_osd & ~ambiguous & ~stopping).sum()),
        }
    else:
        cls_osd = {
            "n_avoid": None,
            "n_ambiguous": None,
            "n_stopping": int((fail_osd & stopping).sum()),
            "n_both": None,
            "n_neither": None,
        }
    cls_bp = {
        "n_avoid": None if avoid_bp is None else int(avoid_bp.sum()),
        "n_non_convergence": int((fail_bp & ~valid_bp).sum()),
        "n_wrong_convergence": int((fail_bp & valid_bp).sum()),
        "wrong_convergence_ambiguous": (
            None if avoid_bp is None
            else int((fail_bp & valid_bp & ambiguous).sum())),
        "wrong_convergence_stopping": int(
            (fail_bp & valid_bp & stopping).sum()),
    }

    unique_keys = sorted({int(k) for k in _syndrome_keys(syndromes)})
    lp = None
    if cfg.lp:
        labels = avoid_osd if use_oracle else fail_osd
        weights = np.ones(cfg.shots)
        lp, frac_by_key = _lp_calibration(h, syndromes, labels, weights, cfg)
        lp["label"] = "avoidable" if use_oracle else "failure"
        lp["n_ambiguous_syn"] = None
        lp["n_unambiguous_syn"] = None
        if use_oracle:
            n_frac_ambig = 0
            n_frac_unambig = 0
            for k, v in frac_by_key.items():
                if v is None or not v:
                    continue
                a = oracle.is_ambiguous(unpack_bits(k, h.shape[0]))
                if a == "timeout":
                    continue
                if a:
                    n_frac_ambig += 1
                else:
                    n_frac_unambig += 1
            lp["n_frac_ambiguous_syn"] = n_frac_ambig
            lp["n_frac_unambiguous_syn"] = n_frac_unambig
        else:
            lp["n_frac_ambiguous_syn"] = None
            lp["n_frac_unambiguous_syn"] = None

    meta = {
        "name": name,
        "sampled": True,
        "n": n,
        "m_x": int(h_x.shape[0]),
        "m_z": int(h.shape[0]),
        "k": int(code_info(h_x, h_z)),
        "d": d_analytic,
        "n_patterns": None,
        "shots": cfg.shots,
        "n_unique_syndromes": len(unique_keys),
        "n_degenerate_syndromes": None,
        "n_ambiguous_syndromes": None,
        "max_minimal_multiplicity": None,
        "n_invalid_bp": int((~valid_bp).sum()),
        "n_invalid_osd": int((~valid_osd).sum()),
        "n_minimal_gt1_instances": None,
        "p": cfg.p,
        "seed": cfg.seed,
        "oracle": use_oracle,
        "device": device,
        "oracle_solves": None if oracle is None else oracle.n_solves,
        "oracle_timeouts": None if oracle is None else oracle.n_timeouts,
    }
    return {
        "meta": meta,
        "per_weight": None,
        "weighted": weighted,
        "classification_bp_osd": cls_osd,
        "classification_bp": cls_bp,
        "lp": lp,
    }


def run_study(cfg=None):
    """Run the study over cfg.codes and attach the claim sentence."""
    cfg = cfg or StudyConfig()
    registry = code_registry()
    codes = {}
    for name in cfg.codes:
        builder, d_analytic = registry[name]
        h_x, h_z = builder()
        codes[name] = run_code(name, h_x, h_z, cfg, d_analytic=d_analytic)
    results = {
        "config": asdict(cfg),
        "codes": codes,
        "families": {
            "degenerate": [c for c in cfg.codes if c.startswith("hgp")],
            "bicycle": [c for c in cfg.codes if c.startswith("bb")],
            "controls": [c for c in cfg.codes
                         if c in ("steane", "rep5")],
        },
    }
    results["claim"] = make_claim(results)
    return results
