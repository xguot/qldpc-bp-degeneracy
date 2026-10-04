"""Run the failure-mode study and write results plus the report.

Usage: python scripts/run_study.py [--p 0.1] [--no-lp] [--out results]

Writes study.json and claim.json under the output directory and
REPORT.md next to them.
"""

import argparse
import json
from pathlib import Path

from failuremodes.report import render_report
from failuremodes.study import StudyConfig, run_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--no-lp", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    cfg = StudyConfig(p=args.p, lp=not args.no_lp)
    results = run_study(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "study.json").write_text(json.dumps(results, indent=2))
    (out / "claim.json").write_text(json.dumps(results["claim"], indent=2))
    (out / "REPORT.md").write_text(render_report(results))
    print(f"wrote {out / 'study.json'}, {out / 'claim.json'}, "
          f"{out / 'REPORT.md'}")
    print(results["claim"]["sentence"])


if __name__ == "__main__":
    main()
