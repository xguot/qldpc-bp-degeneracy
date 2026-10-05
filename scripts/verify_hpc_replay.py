"""Verify the CUDA HPC re-run: nested device, invalid OSD, oracle timeouts, metrics."""
import json
import glob

rows = []
for f in sorted(glob.glob("results/hpc/*/study.json")):
    d = json.load(open(f))
    cfg = d["config"]
    rows.append((f, cfg, d["codes"], d.get("claim", {})))

bad_device = []
bad_osd = []
bad_oracle = []
for f, cfg, codes, claim in rows:
    if cfg.get("device") != "cuda":
        bad_device.append(f)
    for code, cd in codes.items():
        meta = cd.get("meta", {})
        if meta.get("device") != "cuda":
            bad_device.append(f)
        if meta.get("n_invalid_osd", 0) != 0:
            bad_osd.append((f, code, meta.get("n_invalid_osd")))
        if meta.get("oracle_timeouts", 0) not in (0, None):
            bad_oracle.append((f, code, meta.get("oracle_timeouts")))

print("total runs:", len(rows))
print("device!=cuda:", bad_device if bad_device else "none")
print("n_invalid_osd nonzero:", bad_osd if bad_osd else "none")
print("oracle_timeouts nonzero:", bad_oracle if bad_oracle else "none")
print("---")
def fmt(x, nd=4):
    return "   None" if x is None else f"{x:{nd}.{nd}f}"

for f, cfg, codes, claim in rows:
    tag = f.split("/")[-2]
    w = list(codes.values())[0]["weighted"][0]
    amb = claim.get("ambiguous_share_avoidable_bposd")
    lp_prec = claim.get("lp_precision")
    print(
        f"{tag:16s} shots={w['shots']:6d} fail_bp={fmt(w.get('fail_bp'))} "
        f"fail_osd={fmt(w.get('fail_osd'))} avoid_osd={fmt(w.get('avoid_osd'))} "
        f"amb={fmt(amb, 3)} lp_prec={fmt(lp_prec, 3)}"
    )
