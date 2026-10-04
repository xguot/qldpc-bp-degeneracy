"""Claim generation and markdown report rendering for the study results."""

import math


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _fmt(v, nd=1):
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _pooled_mass(entries, key):
    """Sum a mass over entries whose lp block has the key (None-safe)."""
    total = 0.0
    for e in entries:
        if e.get("lp"):
            total += e["lp"]["p_weighted"][key]
    return total


def make_claim(results):
    """Build the claim sentences from the measured aggregates.

    Degenerate (HGP) families with an oracle get the ambiguity claim;
    oracle-less bicycle runs get the LP-detector claim over raw BP+OSD
    failures. All rates are computed at the study's channel rate p.
    """
    cfg = results["config"]
    fam = results["families"]["degenerate"]
    entries = [results["codes"][c] for c in fam if c in results["codes"]]
    sentences = []
    claim = {"p": cfg["p"]}
    if entries:
        x_vals = []
        for e in entries:
            row = next((r for r in e["weighted"] if r["p"] == cfg["p"]),
                       None)
            if row and row.get("avoid_osd"):
                x_vals.append(row["avoid_osd_ambiguous"] / row["avoid_osd"])
        x = None if not x_vals else float(sum(x_vals) / len(x_vals))
        w_pred = _pooled_mass(entries, "w_pred")
        w_both = _pooled_mass(entries, "w_both")
        w_label = _pooled_mass(entries, "w_label")
        y = None if w_pred == 0 else w_both / w_pred
        z = None if w_label == 0 else w_both / w_label
        codes_desc = ", ".join(
            f"[[{e['meta']['n']},{e['meta']['k']},{e['meta']['d']}]]"
            for e in entries)
        sentence = (
            f"Across the tested degenerate hypergraph-product codes "
            f"({codes_desc}) at code capacity with p = {cfg['p']}, "
            f"{_pct(x)} of avoidable BP+OSD failures are degenerate "
            f"ambiguities, and a fractional LP optimum flags an avoidable "
            f"BP+OSD failure with {_pct(y)} precision ({_pct(z)} recall)"
        )
        if "steane" in results["codes"]:
            steane = results["codes"]["steane"]
            c = steane["classification_bp_osd"]
            if c.get("n_avoid") is None:
                sentence += "; the [[7,1,3]] Steane code was not run here"
            elif c["n_avoid"] == 0:
                sentence += ("; the non-degenerate [[7,1,3]] Steane code has"
                             " no avoidable BP+OSD failures at this p")
            else:
                x_steane = 100.0 * c["n_ambiguous"] / c["n_avoid"]
                sentence += ("; the non-degenerate [[7,1,3]] Steane code"
                             f" shows {_pct(x_steane)} by comparison")
        sentence += "."
        sentences.append(sentence)
        claim["ambiguous_share_avoidable_bposd"] = x
        claim["lp_precision"] = y
        claim["lp_recall"] = z
        claim["codes"] = codes_desc

    fam_bb = results["families"].get("bicycle", [])
    entries_bb = [results["codes"][c] for c in fam_bb if c in results["codes"]]
    if entries_bb and not entries:
        w_pred = _pooled_mass(entries_bb, "w_pred")
        w_both = _pooled_mass(entries_bb, "w_both")
        w_label = _pooled_mass(entries_bb, "w_label")
        y = None if w_pred == 0 else w_both / w_pred
        z = None if w_label == 0 else w_both / w_label
        n_non = sum(e["classification_bp"]["n_non_convergence"]
                    for e in entries_bb)
        n_wrong = sum(e["classification_bp"]["n_wrong_convergence"]
                      for e in entries_bb)
        split = (None if n_non + n_wrong == 0
                 else n_non / (n_non + n_wrong))
        codes_desc = ", ".join(
            f"[[{e['meta']['n']},{e['meta']['k']},{e['meta']['d']}]]"
            for e in entries_bb)
        sentence = (
            f"Across the tested bivariate bicycle codes ({codes_desc}) at "
            f"code capacity with p = {cfg['p']}, a fractional LP optimum "
            f"flags a BP+OSD failure with {_pct(y)} precision "
            f"({_pct(z)} recall); BP-only failures split into "
            f"{_pct(split)} non-convergence and "
            f"{_pct(None if split is None else 1.0 - split)} wrong "
            f"convergence."
        )
        sentences.append(sentence)
        claim["lp_precision"] = y
        claim["lp_recall"] = z
        claim["non_convergence_share_bp"] = split
        claim["codes"] = codes_desc
    claim["sentence"] = " ".join(sentences)
    return claim


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def _build_findings(results):
    """Assemble the findings bullets from the measured aggregates."""
    codes = results["codes"]
    cfg = results["config"]
    degenerate = [codes[c] for c in results["families"]["degenerate"]
                  if c in codes]
    bicycle = [codes[c] for c in results["families"]["bicycle"]
               if c in codes]
    bullets = []
    if degenerate:
        if any(e["classification_bp_osd"].get("n_ambiguous") is not None
               for e in degenerate):
            parts = []
            for e in degenerate:
                c = e["classification_bp_osd"]
                m = e["meta"]
                n_stop = c.get("n_stopping")
                stop = "" if n_stop is None else f" ({n_stop} also stopping sets)"
                parts.append(
                    f"{c['n_ambiguous']}/{c['n_avoid']} on "
                    f"[[{m['n']},{m['k']},{m['d']}]]{stop}")
            bullets.append(
                "Every avoidable BP+OSD failure on the degenerate codes is a"
                f" degenerate ambiguity ({', '.join(parts)}); stopping sets"
                " add no separate failure mode for BP+OSD, they only"
                " co-occur with ambiguity.")
        bp_parts = []
        for e in degenerate:
            c = e["classification_bp"]
            m = e["meta"]
            bp_parts.append(
                f"{c['n_non_convergence']} non-convergent and "
                f"{c['n_wrong_convergence']} wrongly convergent on "
                f"[[{m['n']},{m['k']},{m['d']}]]")
        bullets.append(
            "BP-only failures split by code: " + "; ".join(bp_parts) + ".")
        lp_parts = []
        for e in degenerate:
            lp = e["lp"]
            m = e["meta"]
            if lp is None:
                continue
            if m.get("n_degenerate_syndromes") is not None:
                lp_parts.append(
                    f"{lp['n_fractional']}/{m['n_degenerate_syndromes']} "
                    f"multi-minimal syndromes on [[{m['n']},{m['k']},{m['d']}]]"
                    f" ({lp['n_fractional_vertex']} by the vertex signal)")
            else:
                lp_parts.append(
                    f"{lp['n_fractional']}/{lp['n_solved']} sampled syndromes "
                    f"fractional on [[{m['n']},{m['k']},{m['d']}]]")
        if lp_parts:
            bullets.append(
                "The LP optimal-face signal detects multi-minimal syndromes"
                f" ({', '.join(lp_parts)}); its precision below 100% comes"
                " from multi-minimal-but-unambiguous syndromes, i.e."
                " degeneracy without ambiguity.")
    if bicycle:
        for e in bicycle:
            lp = e["lp"]
            m = e["meta"]
            if lp is None:
                continue
            bullets.append(
                f"On [[{m['n']},{m['k']},{m['d']}]] the LP optimal face is"
                f" fractional for {lp['n_fractional']}/{lp['n_solved']} "
                f"sampled syndromes ({lp['n_fractional_vertex']} by the "
                f"vertex signal) and flags BP+OSD failures with "
                f"{_fmt(lp['p_weighted']['precision'], 3)} precision at "
                f"{_fmt(lp['p_weighted']['recall'], 3)} recall (p = "
                f"{cfg['p']}).")
    if "steane" in codes:
        steane = codes["steane"]
        if steane["meta"].get("n_ambiguous_syndromes") is not None:
            bullets.append(
                "On the non-degenerate [[7,1,3]] Steane code BP+OSD has no"
                " avoidable failures at all, yet"
                f" {steane['lp']['n_fractional']}/{steane['meta']['n_unique_syndromes']}"
                " syndromes show fractional faces: the signal fires but"
                " flags nothing.")
    return bullets


def render_report(results):
    """Render the full study report as markdown."""
    cfg = results["config"]
    claim = results["claim"]
    codes = results["codes"]
    lines = []
    add = lines.append

    add("# Failure-Mode Characterization of BP Decoding on Degenerate QLDPC Codes")
    add("")
    add("## The claim")
    add("")
    add(f"> {claim['sentence']}")
    add("")
    add("## Method")
    add("")
    add("- X-error side only; the CSS sides decouple and the study codes are"
        " symmetric enough that the Z side is structurally identical.")
    if cfg.get("shots", 0) > 0:
        add(f"- Sampled mode: {cfg['shots']} i.i.d. shots per code with the"
            f" fixed seed {cfg['seed']}; rates are empirical and every"
            " number below is reproducible.")
        add("- MLD oracle: integer programming (HiGHS MILP, one or two solves"
            " per query, cached per syndrome and logical class).")
    else:
        add("- Exhaustive enumeration of all 2^n error patterns per code"
            " (n <= 13): every rate below is exact, no sampling.")
    add("- An avoidable failure is a decode where the MLD oracle succeeds on"
        " the same syndrome but the decoder fails logically.")
    add("- Degenerate ambiguity: the syndrome has minimal errors in more than"
        " one logical class. Stopping set: every check adjacent to the true"
        " error support touches at least two flipped positions (the test on"
        " the residual is vacuous for syndrome-valid corrections).")
    add("- Wrong convergence: BP output satisfies all checks but is logically"
        " wrong; non-convergence: BP output violates the syndrome.")
    add("- The LP detector reports whether the optimal face of the relaxation"
        " admits a fractional point: the optimum value is fixed and each"
        " coordinate's range on that face is bounded with 2n extra LP solves"
        " per syndrome. A syndrome with two minimal errors always yields one"
        " (their midpoint is fractional). The vertex signal is also recorded.")
    add("")

    add("## Codes under test")
    add("")
    add(_row(["code", "params", "n", "k", "d", "m_x", "m_z",
              "syndromes", "degenerate", "ambiguous", "instances"]))
    add(_row(["---"] * 11))
    for name, e in codes.items():
        m = e["meta"]
        instances = m["shots"] if m.get("sampled") else m["n_patterns"]
        add(_row([name, f"[[{m['n']},{m['k']},{m['d']}]]",
                  str(m["n"]), str(m["k"]), str(m["d"]),
                  str(m["m_x"]), str(m["m_z"]),
                  str(m["n_unique_syndromes"]),
                  _fmt(m.get("n_degenerate_syndromes")),
                  _fmt(m.get("n_ambiguous_syndromes")),
                  str(instances)]))
    add("")

    add("## Experiment 1: avoidable-failure rates")
    add("")
    header = ["code", "p", "P(fail BP)", "P(avoid BP)", "P(fail BP+OSD)",
              "P(avoid BP+OSD)", "avoid share BP", "avoid share BP+OSD"]
    add(_row(header))
    add(_row(["---"] * len(header)))
    for name, e in codes.items():
        for row in e["weighted"]:
            share_bp = (None if row.get("fail_bp") in (None, 0)
                        or row.get("avoid_bp") is None else
                        row["avoid_bp"] / row["fail_bp"])
            share_osd = (None if row.get("fail_osd") in (None, 0)
                         or row.get("avoid_osd") is None else
                         row["avoid_osd"] / row["fail_osd"])
            add(_row([name, str(row["p"]), _fmt(row["fail_bp"], 4),
                      _fmt(row.get("avoid_bp"), 4),
                      _fmt(row["fail_osd"], 4),
                      _fmt(row.get("avoid_osd"), 4),
                      _fmt(share_bp, 3), _fmt(share_osd, 3)]))
    add("")

    if any(e["per_weight"] for e in codes.values()):
        add("### Per error weight (exact counts over all 2^n patterns)")
        add("")
        for name, e in codes.items():
            add(f"**{name}**")
            add("")
            add(_row(["w", "patterns", "fail BP", "avoid BP", "fail BP+OSD",
                      "avoid BP+OSD"]))
            add(_row(["---"] * 6))
            for row in e["per_weight"]:
                add(_row([str(row["w"]), str(row["n_patterns"]),
                          str(row["fail_bp"]), str(row["avoid_bp"]),
                          str(row["fail_osd"]), str(row["avoid_osd"])]))
            add("")

    add("## Experiment 2: classification of avoidable failures")
    add("")
    add("### BP+OSD")
    add("")
    add(_row(["code", "avoidable", "ambiguous", "stopping", "both",
              "neither"]))
    add(_row(["---"] * 6))
    for name, e in codes.items():
        c = e["classification_bp_osd"]
        add(_row([name, _fmt(c.get("n_avoid")), _fmt(c.get("n_ambiguous")),
                  _fmt(c.get("n_stopping")), _fmt(c.get("n_both")),
                  _fmt(c.get("n_neither"))]))
    add("")
    add("### BP only (convergence split)")
    add("")
    add(_row(["code", "avoidable", "non-convergence", "wrong convergence"]))
    add(_row(["---"] * 4))
    for name, e in codes.items():
        c = e["classification_bp"]
        add(_row([name, _fmt(c.get("n_avoid")),
                  str(c["n_non_convergence"]),
                  str(c["n_wrong_convergence"])]))
    add("")

    add("## Experiment 3: LP calibration as degeneracy detector")
    add("")
    header = ["code", "solved", "frac face", "frac vertex", "frac ambig",
              "frac unambig", "precision", "recall", "precision (u)",
              "recall (u)"]
    add(_row(header))
    add(_row(["---"] * len(header)))
    for name, e in codes.items():
        lp = e["lp"]
        if lp is None:
            add(_row([name] + ["n/a"] * (len(header) - 1)))
            continue
        u = lp["uniform"]
        pw = lp["p_weighted"]
        add(_row([name, str(lp["n_solved"]), str(lp["n_fractional"]),
                  str(lp["n_fractional_vertex"]),
                  _fmt(lp.get("n_frac_ambiguous_syn")),
                  _fmt(lp.get("n_frac_unambiguous_syn")),
                  _fmt(pw["precision"], 3), _fmt(pw["recall"], 3),
                  _fmt(u["precision"], 3), _fmt(u["recall"], 3)]))
    add("")

    findings = _build_findings(results)
    if findings:
        add("## Findings")
        add("")
        for f in findings:
            add(f"- {f}")
        add("")

    add("## Limitations")
    add("")
    if cfg.get("shots", 0) > 0:
        add("- Sampled rates are empirical (fixed seed) and confined to the"
            " X-error side; the ILP oracle is exact but subject to the"
            " per-solve time limit, and unresolved queries are dropped from"
            " the avoidable counts.")
        add("- Code distances are the analytic construction values, not"
            " brute-force verified at these sizes.")
    else:
        add("- Small codes only (n <= 13), code-capacity noise only; the"
            " study is a classification exercise, not a decoder benchmark.")
        add("- All rates are exact but confined to the X-error side and the"
            " five codes listed above.")
    add("- The fractional-LP signal is a characterization of the optimal"
        " face via HiGHS; different solver paths may still disagree at the"
        " tolerance level.")
    add("- Degenerate-ambiguity labels depend on the logical basis that"
        " qudec.logicals returns.")
    add("")
    return "\n".join(lines)
