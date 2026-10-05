"""ADMM parity-polytope decoding study under phenomenological noise.

Runs qudec's batched ADMM decoder (Gu and Soleimanifar, ISIT 2026) with
OSD post-processing on the [[144,12,12]] bivariate bicycle code under
phenomenological noise (per-round data and measurement errors, qudec's
phenom module). Two modes:

  confirm - reference config (rho=2, OSD-CS, plain ADMM) over a
            distance x rate grid at 4000 shots per cell
  ablate  - robustness grid over rho, OSD order, over-relaxation alpha,
            and the LDR subgradient step beta

The decoder config is passed on the command line; the SLURM scripts
compute the grid cells from the array task id. Each task writes one
JSON to results/admm_bb144/<tag>.json; scripts/aggregate_admm.py
merges them.

Measurement errors use rate p_meas = p (sample_phenom's default), the
standard phenomenological-noise setting.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import torch
    from qudec.admm import AdmmOsdDecoder, admm_solve_batch
    from qudec.codes import gross_code, medium_code, steane_code, logicals
    from qudec.phenom import PhenomDecoder, benchmark_phenom
except ImportError as exc:
    sys.exit(
        f"missing qudec module ({exc}). On Rivanna, run bash hpc/setup.sh "
        "again so qudec is updated with admm.py and phenom.py."
    )

CODES = {
    "bb144": gross_code,
    "bb72": medium_code,
    "steane": steane_code,
}


class AlphaAdmmOsdDecoder(AdmmOsdDecoder):
    """AdmmOsdDecoder with an exposed over-relaxation alpha.

    qudec's decoder hardcodes alpha = 1.0 in the ADMM y-update; this
    subclass routes the parameter through for the plain (non-LDR) path,
    which is the only path where alpha applies. With alpha = 1.0 the
    output equals the base decoder exactly (unit-tested).
    """

    def __init__(self, *args, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        if self.ldr and self.alpha != 1.0:
            raise ValueError("alpha only applies to the plain ADMM path")

    def decode_corrections(self, sx, sz):
        kw = dict(rho=self.rho, max_iter=self.max_iter,
                  max_r=self.max_r, alpha=self.alpha)
        x_x = admm_solve_batch(self.h_z, sx, c_vec=self.weights_x,
                               **kw).cpu().numpy()
        x_z = admm_solve_batch(self.h_x, sz, c_vec=self.weights_z,
                               **kw).cpu().numpy()
        c_x = np.stack([self._decode_osd(self.h_z, sx[i], x_x[i])
                        for i in range(sx.shape[0])], axis=0)
        c_z = np.stack([self._decode_osd(self.h_x, sz[i], x_z[i])
                        for i in range(sz.shape[0])], axis=0)
        return c_x, c_z


def run(cfg):
    h_x, h_z = CODES[cfg["code"]]()
    l_x, l_z = logicals(h_x, h_z)
    kwargs = dict(rho=cfg["rho"], osd_order=cfg["osd_order"])
    if cfg["ldr"]:
        kwargs.update(ldr=True, ldr_outer=cfg["ldr_outer"],
                      ldr_beta=cfg["ldr_beta"])
    decoder_cls = AdmmOsdDecoder if cfg["alpha"] == 1.0 \
        else AlphaAdmmOsdDecoder
    if decoder_cls is AlphaAdmmOsdDecoder:
        kwargs["alpha"] = cfg["alpha"]
    decoder = PhenomDecoder(decoder_cls, h_x, h_z, l_x, l_z, cfg["d"],
                            **kwargs)
    t0 = time.time()
    res = benchmark_phenom(decoder, h_x, h_z, l_x, l_z, cfg["p"], cfg["d"],
                           cfg["shots"], model="x", seed=cfg["seed"])
    res.update({
        "code": cfg["code"],
        "d": cfg["d"],
        "rho": cfg["rho"],
        "osd_order": cfg["osd_order"],
        "alpha": cfg["alpha"],
        "ldr": cfg["ldr"],
        "ldr_outer": cfg.get("ldr_outer", 5),
        "ldr_beta": cfg.get("ldr_beta", 1.0),
        "seed": cfg["seed"],
        "wall_seconds": round(time.time() - t0, 1),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["confirm", "ablate"], required=True)
    ap.add_argument("--config-id", type=int, required=True,
                    help="SLURM_ARRAY_TASK_ID")
    ap.add_argument("--code", default="bb144")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--shots", type=int, default=2500)
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--osd-order", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--ldr", type=int, default=0)
    ap.add_argument("--ldr-outer", type=int, default=5)
    ap.add_argument("--ldr-beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="results/admm_bb144")
    args = ap.parse_args()

    cfg = {
        "mode": args.mode,
        "config_id": args.config_id,
        "code": args.code,
        "p": args.p,
        "d": args.d,
        "shots": args.shots,
        "rho": args.rho,
        "osd_order": args.osd_order,
        "alpha": args.alpha,
        "ldr": bool(args.ldr),
        "ldr_outer": args.ldr_outer,
        "ldr_beta": args.ldr_beta,
        "seed": args.seed,
    }
    tag = f"{args.mode}_{args.config_id}"
    res = run(cfg)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{tag}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"=== wrote {args.out}/{tag}.json: "
          f"ler={res['ler']:.4f} invalid={res['invalid_x']}"
          f"+{res['invalid_z']} wall={res['wall_seconds']}s "
          f"device={res['device']} ===")


if __name__ == "__main__":
    main()
