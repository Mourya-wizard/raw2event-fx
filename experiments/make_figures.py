"""Results dashboard: Hero results figure for the README, built ONLY from results/*.csv (no new measurements) -> results/hero.png.
Usage: python -m experiments.make_figures
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots 1-3 (dataviz reference palette, light mode); identity is fixed across panels.
RAW, ISP, FIX = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})


def bars(ax, groups, series, ylabel):
    """Grouped bars with direct value labels; series = [(label, color, values, stds_or_None, hatch)]."""
    n, x = len(series), np.arange(len(groups))
    wd = 0.8 / n
    for i, (lbl, col, v, sd, hatch) in enumerate(series):
        pos = x - 0.4 + wd * (i + 0.5)
        ax.bar(pos, v, wd - 0.03, color=col, label=lbl, hatch=hatch, edgecolor=SURF if not hatch else "white", linewidth=0)
        if sd is not None:
            ax.errorbar(pos, v, sd, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        for p, val, s in zip(pos, v, sd if sd is not None else [0] * len(v)):
            ax.text(p, val + s + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)


def title(ax, t, sub):
    ax.set_title(t, loc="left", fontsize=12, fontweight="bold", color=INK, pad=18)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")


fig, ax = plt.subplots(2, 2, figsize=(14, 9.2))

# (a) M2: bit-width Pareto, interp on, F1 mean vs cost
m2 = pd.read_csv("results/m2_summary.csv")
d = m2[m2.interp == 1]
a = ax[0, 0]
for lb, mk in [(12, "s"), (10, "D"), (8, "o"), (6, "^")]:
    e = d[d.lut_bits == lb]
    a.plot(e.cost, e.f1_mean, mk + "-", color=FIX, alpha=0.35 + 0.65 * (lb == 8), ms=7, lw=1.5, label=f"LUT_BITS={lb}")
best = d[(d.fl == 10) & (d.lut_bits == 8)].iloc[0]
a.scatter([best.cost], [best.f1_mean], s=260, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=5)
a.annotate(f"chosen: FL=10, LUT=8\nF1 {best.f1_mean:.4f} @ {int(best.cost)} bits", (best.cost, best.f1_mean),
           xytext=(best.cost * 2.2, 0.93), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.axhline(0.99, ls="--", c=INK2, lw=0.8); a.text(560, 0.9915, "F1 = 0.99", fontsize=8, color=INK2)
a.set_xscale("log"); a.set_xlabel("hardware cost proxy (bits, log)"); a.set_ylabel("F1 vs float reference")
a.legend(fontsize=8, frameon=False, loc="lower right")
title(a, "M2 · Fixed-point bit-width study", "interp on; markers = FL 4…16 along each curve; mean over 3 scenes")

# (b) M3: F1 vs 1000 fps ground truth, noise off
m3 = pd.read_csv("results/m3_ablation.csv")
m3 = m3[m3.noise == "off"]
sc = ["texture", "checker", "bars"]
get = lambda pfx: [m3[(m3.scene == s) & m3.path.str.startswith(pfx)].f1.iloc[0] for s in sc]
bars(ax[0, 1], sc, [("float raw-first", RAW, get("float raw-first"), None, None),
                    ("float ISP-first", ISP, get("float ISP-first"), None, None),
                    ("ISP-first, C/2.4", ISP, get("float ISP-first, C"), None, "///"),
                    ("fixed raw-first", FIX, get("fixed"), None, None)], "F1 vs ground truth")
ax[0, 1].set_ylim(0, 1.12); ax[0, 1].legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
title(ax[0, 1], "M3 · Raw-first vs ISP-first (synthetic)", "30 fps camera vs 1000 fps ground-truth events, noise off, same C=0.25")

# (c) M5: task accuracy vs FL
m5 = pd.read_csv("results/m5_task.csv")
c = ax[1, 0]
fx = m5[(m5.train == m5.test) & m5.test.str.startswith("fixed")].assign(fl=lambda q: q.test.str[8:].astype(int)).sort_values("fl")
c.errorbar(fx.fl, fx["mean"], fx["std"], fmt="o-", color=FIX, ms=8, lw=2, capsize=3, label="fixed raw-first (accuracy)")
c.plot(fx.fl, fx.mnist_event_f1_vs_float, "s--", color=FIX, alpha=0.45, ms=6, lw=1.5, label="fixed vs float (event F1)")
for name, col in [("float raw-first", RAW), ("float ISP-first", ISP)]:
    r = m5[(m5.train == name) & (m5.test == name)].iloc[0]
    c.axhline(r["mean"], color=col, lw=2, label=f"{name} {r['mean']:.3f}")
c.set_xlabel("fractional bits FL"); c.set_ylabel("MNIST accuracy / event F1"); c.set_ylim(0.7, 1.0)
c.legend(fontsize=8, frameon=False, loc="lower right")
title(c, "M5 · Downstream task (MNIST, 3 seeds)", "accuracy flat down to FL=4 while event F1 drops to 0.75")

# (d) M6 + M7: real DAVIS346 events
m6 = pd.read_csv("results/m6_summary.csv").set_index("path")
m7 = pd.read_csv("results/m7_summary.csv").set_index("path")
pick = lambda df, col, pfx: df[df.index.str.startswith(pfx)][col].iloc[0]
grp = ["CED · same-pixel\n(7 bags, r=0)", "Raw2Event · cross-camera\n(15 recs, r=0)", "Raw2Event · cross-camera\n(15 recs, r=2 px)"]
ser = []
for lbl, col, pfx in [("float raw-first", RAW, "float raw"), ("float ISP-first", ISP, "float ISP"), ("fixed raw-first", FIX, "fixed")]:
    ser.append((lbl, col, [pick(m6, "f1_mean", pfx), pick(m7, "f1_r0_mean", pfx), pick(m7, "f1_r2_mean", pfx)],
                [pick(m6, "f1_std", pfx), pick(m7, "f1_r0_std", pfx), pick(m7, "f1_r2_std", pfx)], None))
bars(ax[1, 1], grp, ser, "F1 vs REAL DAVIS346 events")
ax[1, 1].set_ylim(0, 0.66); ax[1, 1].legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
title(ax[1, 1], "M6 · M7 · Validation on real event cameras", "C calibrated per path on held-out recordings; mean ± std")

plt.tight_layout(h_pad=3.2, w_pad=2.5)
plt.savefig("results/hero.png", dpi=130)
print("saved results/hero.png")
