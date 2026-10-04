"""Aggregate per-run study.json files under results/hpc into one summary.

Usage: python scripts/aggregate_hpc.py [results/hpc]

Writes hpc_summary.md and hpc_summary.json next to the input directory.
"""

import argparse
import json
from pathlib import Path

HEADERS = [
    ("code", "code"),
    ("p", "p"),
    ("shots", "shots"),
    ("fail_bp", "P(fail BP)"),
    ("avoid_bp", "P(avoid BP)"),
    ("fail_osd", "P(fail BP+OSD)"),
    ("avoid_osd", "P(avoid BP+OSD)"),
    ("ambig_share", "ambig share"),
    ("stop_share", "stopping share"),
    ("non_conv", "BP non-conv"),
    ("wrong_conv", "BP wrong-conv"),
    ("lp_prec", "LP precision"),
    ("lp_rec", "LP recall"),
    ("frac_face", "LP frac face"),
    ("frac_vertex", "LP frac vertex"),
    ("timeouts", "ILP timeouts"),
]


def load_runs(root):
    rows = []
    for path in sorted(Path(root).glob("*/study.json")):
        r = json.loads(path.read_text())
        for name, e in r["codes"].items():
            m = e["meta"]
            row = {"code": name, "p": m["p"], "shots": m["shots"],
                   "timeouts": m.get("oracle_timeouts")}
            weighted = e["weighted"][0]
            for key in ("fail_bp", "avoid_bp", "fail_osd", "avoid_osd"):
                row[key] = weighted.get(key)
            cls = e["classification_bp_osd"]
            cls_bp = e["classification_bp"]
            n_avoid = cls.get("n_avoid")
            if n_avoid:
                row["ambig_share"] = cls["n_ambiguous"] / n_avoid
                row["stop_share"] = cls["n_stopping"] / n_avoid
            else:
                row["ambig_share"] = None
                row["stop_share"] = None
            row["non_conv"] = cls_bp["n_non_convergence"]
            row["wrong_conv"] = cls_bp["n_wrong_convergence"]
            if e["lp"] is not None:
                row["lp_prec"] = e["lp"]["p_weighted"]["precision"]
                row["lp_rec"] = e["lp"]["p_weighted"]["recall"]
                row["frac_face"] = e["lp"]["n_fractional"]
                row["frac_vertex"] = e["lp"]["n_fractional_vertex"]
            else:
                row["lp_prec"] = row["lp_rec"] = None
                row["frac_face"] = row["frac_vertex"] = None
            rows.append(row)
    rows.sort(key=lambda r: (r["code"], r["p"]))
    return rows


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def render(rows):
    lines = ["# HPC study summary", ""]
    lines.append("| " + " | ".join(h[1] for h in HEADERS) + " |")
    lines.append("|" + "---|" * len(HEADERS))
    for r in rows:
        cells = []
        for key, _ in HEADERS:
            if key in ("code", "p", "shots", "non_conv", "wrong_conv",
                       "frac_face", "frac_vertex", "timeouts"):
                cells.append(str(r[key]))
            else:
                cells.append(fmt(r[key]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="results/hpc")
    args = parser.parse_args(argv)
    root = Path(args.root)
    rows = load_runs(root)
    if not rows:
        raise SystemExit(f"no results/hpc/*/study.json found under {root}")
    (root / "hpc_summary.json").write_text(
        json.dumps(rows, indent=2))
    (root / "hpc_summary.md").write_text(render(rows))
    print(f"wrote {root / 'hpc_summary.json'} and "
          f"{root / 'hpc_summary.md'} from {len(rows)} runs")


if __name__ == "__main__":
    main()
