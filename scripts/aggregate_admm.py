"""Aggregate the ADMM study JSONs into one summary table.

Reads results/admm_bb144/confirm_*.json and ablate_*.json and writes
results/admm_bb144/summary.md.
"""

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "admm_bb144"


def load_all(pattern):
    out = []
    for f in sorted(glob.glob(str(DIR / pattern))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def main():
    conf = load_all("confirm_*.json")
    abl = load_all("ablate_*.json")
    lines = ["# ADMM study summary (bb144, phenomenological noise)",
             ""]
    if conf:
        lines += ["## Confirmation: reference ADMM (rho=2, OSD-CS), 4000 shots",
                  "",
                  "| d | p | LER | invalid | wall (s) |",
                  "|---|---|---|---|---|"]
        for r in sorted(conf, key=lambda x: (x["d"], x["p"])):
            lines.append(
                f"| {r['d']} | {r['p']} | {r['ler']:.4f} | "
                f"{r['invalid_x']}+{r['invalid_z']} | {r['wall_seconds']} |")
    if abl:
        lines += ["", "## Ablation: 15 configs x 2 p, d=3, 2500 shots",
                  "",
                  "| p | rho | osd | alpha | ldr | beta | LER | invalid |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sorted(abl, key=lambda x: (x["p"], x["rho"], x["osd_order"],
                                            x["alpha"], x["ldr_beta"])):
            ldr = "yes" if r["ldr"] else ""
            beta = f"{r['ldr_beta']}" if r["ldr"] else ""
            lines.append(
                f"| {r['p']} | {r['rho']} | {r['osd_order']} | "
                f"{r['alpha']} | {ldr} | {beta} | {r['ler']:.4f} | "
                f"{r['invalid_x']}+{r['invalid_z']} |")
    text = "\n".join(lines) + "\n"
    (DIR / "summary.md").write_text(text)
    print(f"wrote {DIR / 'summary.md'} from "
          f"{len(conf)} confirm and {len(abl)} ablate runs")


if __name__ == "__main__":
    main()
