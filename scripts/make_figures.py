"""Generate the figures for the research note from the committed results.

Reads results/study.json (exhaustive) and results/hpc/<code>_p<rate>/
study.json (sampled) and writes the four note figures to docs/figs.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

P_TAGS = {"0.02": "0_02", "0.05": "0_05", "0.08": "0_08", "0.10": "0_10"}


def load(path):
    with open(path) as f:
        return json.load(f)


def exhaustive_entry(code):
    r = load(ROOT / "results" / "study.json")
    return r["codes"][code]


def hpc_entry(code, p):
    path = ROOT / "results" / "hpc" / f"{code}_p{P_TAGS[p]}" / "study.json"
    r = load(path)
    return list(r["codes"].values())[0]


def weighted_row(entry, p):
    for row in entry["weighted"]:
        if row["p"] == float(p):
            return row
    raise KeyError(p)


def fig_precision_vs_distance():
    d_points = {"hgp22": 2, "hgp32": 2, "hgp33": 3}
    xs, ys, labels = [], [], []
    for code, d in d_points.items():
        e = exhaustive_entry(code)
        xs.append(d)
        ys.append(100 * e["lp"]["p_weighted"]["precision"])
        labels.append(code)
    for code, d in (("hgp44", 4), ("hgp55", 5), ("hgp66", 6)):
        e = hpc_entry(code, "0.10")
        xs.append(d)
        ys.append(100 * e["lp"]["p_weighted"]["precision"])
        labels.append(code)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(xs[:3], ys[:3], "o", color="tab:blue", label="exhaustive (n <= 13)")
    ax.plot(xs[3:], ys[3:], "s", color="tab:red", label="sampled, 10^4 shots")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("code distance d")
    ax.set_ylabel("LP detection precision (%)")
    ax.set_ylim(0, 60)
    ax.legend(fontsize=8)
    ax.set_title("Fractional-LP precision decays with distance "
                 "(recall 100% everywhere, p = 0.1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_precision_distance.png", dpi=200)
    plt.close(fig)


def fig_avoidable_share():
    codes = ["hgp22", "hgp32", "hgp33", "hgp44", "hgp55", "hgp66"]
    nonavoid, avoid = [], []
    for code in codes:
        if code in ("hgp22", "hgp32", "hgp33"):
            row = weighted_row(exhaustive_entry(code), "0.10")
        else:
            row = weighted_row(hpc_entry(code, "0.10"), "0.10")
        avoid.append(100 * row["avoid_osd"])
        nonavoid.append(100 * (row["fail_osd"] - row["avoid_osd"]))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(codes, nonavoid, label="unavoidable (MLD also fails)")
    ax.bar(codes, avoid, bottom=nonavoid,
           label="avoidable (oracle decodes it)")
    ax.set_ylabel("BP+OSD failure rate (%)")
    ax.set_title("About half of BP+OSD failures on the larger patches "
                 "are avoidable (p = 0.1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_avoidable_share.png", dpi=200)
    plt.close(fig)


def fig_wrong_convergence():
    codes = ["hgp44", "hgp55", "hgp66", "bb72", "bb144"]
    share = []
    for code in codes:
        e = hpc_entry(code, "0.10")
        c = e["classification_bp"]
        total = c["n_non_convergence"] + c["n_wrong_convergence"]
        share.append(100 * c["n_wrong_convergence"] / total)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.bar(codes, share)
    for bar, v in zip(bars, share):
        ax.annotate(f"{v:.0f}%", (bar.get_x() + bar.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("wrong-convergence share of BP failures (%)")
    ax.set_title("BP failure mode: [[144,12,12]] almost never "
                 "wrong-converges (p = 0.1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_wrong_convergence.png", dpi=200)
    plt.close(fig)


def fig_bb_precision():
    ps = ["0.02", "0.05", "0.08", "0.10"]
    for code, color, marker in (("bb72", "tab:blue", "o"),
                                ("bb144", "tab:red", "s")):
        prec = []
        for p in ps:
            e = hpc_entry(code, p)
            prec.append(100 * e["lp"]["p_weighted"]["precision"])
        plt.plot([float(p) for p in ps], prec, marker + "-",
                 color=color, label=code)
    plt.xlabel("channel rate p")
    plt.ylabel("LP detection precision (%)")
    plt.title("Bicycle codes: fractional-LP precision rises with p")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig4_bb_precision.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    fig_precision_vs_distance()
    fig_avoidable_share()
    fig_wrong_convergence()
    fig_bb_precision()
    print(f"wrote figures to {OUT}")
