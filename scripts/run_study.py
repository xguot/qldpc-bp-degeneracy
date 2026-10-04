"""Run the failure-mode study and write results plus the report.

Usage:
  python scripts/run_study.py                                # exhaustive default
  python scripts/run_study.py --codes hgp44 --p 0.05 --shots 10000 --seed 1
  python scripts/run_study.py --codes bb72 --p 0.08 --shots 10000 --no-oracle

Writes study.json and claim.json under the output directory and
REPORT.md next to them.
"""

import argparse
import json
from pathlib import Path

from failuremodes.report import render_report
from failuremodes.study import StudyConfig, code_registry, run_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", default=None,
                        help="comma-separated code names, default: the five "
                             "small exhaustive codes")
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--shots", type=int, default=0,
                        help="shots per code; 0 selects exhaustive mode")
    parser.add_argument("--no-oracle", action="store_true",
                        help="skip the ILP MLD oracle in sampled mode")
    parser.add_argument("--no-lp", action="store_true")
    parser.add_argument("--lp-max-syndromes", type=int, default=4096)
    parser.add_argument("--ilp-time-limit", type=float, default=60.0)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="decoder device for the sampled path")
    parser.add_argument("--gpu-chunk", type=int, default=10000,
                        help="batch chunk size for the GPU decoders")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    registry = code_registry()
    names = None
    if args.codes is not None:
        names = tuple(c.strip() for c in args.codes.split(","))
        for name in names:
            if name not in registry:
                raise SystemExit(f"unknown code {name}, known: "
                                 f"{', '.join(registry)}")
    kwargs = dict(
        p=args.p,
        shots=args.shots,
        oracle="none" if args.no_oracle else "ilp",
        lp=not args.no_lp,
        lp_max_syndromes=args.lp_max_syndromes,
        ilp_time_limit=args.ilp_time_limit,
        device=args.device,
        gpu_chunk=args.gpu_chunk,
        seed=args.seed,
    )
    if names is not None:
        kwargs["codes"] = names
    cfg = StudyConfig(**kwargs)
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
