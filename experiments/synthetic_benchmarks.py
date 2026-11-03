"""Synthetic benchmarks driving the C++ generator: bit-width sweep (M2), raw-first vs ISP-first ablation (M3), latency (M4).
Usage: python -m experiments.synthetic_benchmarks --stage bitwidth ablation latency
Outputs: results/m2_sweep.csv, results/m2_summary.csv, results/m2_pareto.png, results/m3_ablation.csv, results/m4_latency.csv
"""
import os, sys, argparse, subprocess, platform
for k, v in {"XDG_CACHE_HOME": "./.cache", "MPLCONFIGDIR": "./.cache/mpl",
             "HF_HOME": "./.cache/huggingface", "TORCH_HOME": "./.cache/torch"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
from raw2event.scenes import read_frames, SCENES, DATA
from raw2event.events import raw_first_L, isp_first_L, gen_events, C0
from raw2event import metrics as M
from raw2event.native import EXE, RAWDIR, run_cpp, best_cfg

TOL = 0.5 * 1e6 / 30  # us: half the 30 fps camera frame period
IL = 4  # integer bits: ln(cell sum) in [0, ln 4095 = 8.32] fits unsigned Q4.FL


def cost(fl, lb, interp):
    word = IL + fl
    entries = (1 << lb) + (1 if interp else 0)
    return word, entries * word, word + entries * word  # state bits/px, LUT bits, proxy = state + LUT


def score(pred, ref, h, w, t):
    p, r, f1 = M.prf(pred, ref, w, TOL)
    return dict(n_pred=len(pred), n_ref=len(ref), precision=p, recall=r, f1=f1,
                count_rmse=M.count_rmse(pred, ref, h, w), ssim=M.frames_ssim(pred, ref, h, w, t))


def m2():
    rows = []
    for s in SCENES:
        f = f"{DATA}/{s}_30.bin"
        raw, t = read_frames(f)
        h, w = raw.shape[1] // 2, raw.shape[2] // 2
        ref = gen_events(raw_first_L(raw), t)
        for fl in [16, 14, 12, 10, 8, 6, 4]:
            for lb in [12, 10, 8, 6]:
                for it in [1, 0]:
                    ev, _ = run_cpp(f, fl, lb, it, f"{RAWDIR}/m2_{s}.bin")
                    sb, lbits, c = cost(fl, lb, it)
                    rows.append(dict(scene=s, fl=fl, lut_bits=lb, interp=it, state_bits=sb, lut_total_bits=lbits,
                                     cost=c, **score(ev, ref, h, w, t)))
                    print(f"M2 {s:8s} FL={fl:2d} LUT={lb:2d} interp={it} F1={rows[-1]['f1']:.4f} n={len(ev)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/m2_sweep.csv", index=False)
    g = df.groupby(["fl", "lut_bits", "interp", "state_bits", "lut_total_bits", "cost"], as_index=False)
    summ = g.agg(f1_mean=("f1", "mean"), f1_min=("f1", "min"), rmse_mean=("count_rmse", "mean"), ssim_mean=("ssim", "mean"))
    summ = summ.sort_values("cost")
    summ.to_csv("results/m2_summary.csv", index=False)
    ok = summ[summ.f1_min >= 0.99]
    best = ok.iloc[0] if len(ok) else None
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("BEST (smallest cost with min-over-scenes F1 >= 0.99):",
          "NONE" if best is None else f"FL={int(best.fl)} LUT_BITS={int(best.lut_bits)} interp={int(best.interp)} "
          f"cost={int(best.cost)} F1_min={best.f1_min:.4f} F1_mean={best.f1_mean:.4f}")
    pareto_plot(summ, best)


def pareto_plot(summ, best):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for it, mk in [(1, "o"), (0, "x")]:
        d = summ[summ.interp == it]
        sc = ax.scatter(d.cost, d.f1_mean, c=d.fl, marker=mk, cmap="viridis", label=f"interp={it}")
    front, top = [], -1
    for _, r in summ.sort_values("cost").iterrows():
        if r.f1_mean > top:
            front.append(r); top = r.f1_mean
    front = pd.DataFrame(front)
    ax.plot(front.cost, front.f1_mean, "r-", lw=1, label="Pareto front")
    if best is not None:
        ax.scatter([best.cost], [best.f1_mean], s=150, facecolors="none", edgecolors="r", label="chosen")
    ax.axhline(0.99, ls="--", c="gray", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("cost proxy = state bits/px + LUT bits x entries")
    ax.set_ylabel("F1 vs float ref (mean of 3 scenes)"); ax.legend(fontsize=8)
    plt.colorbar(sc, label="FL bits"); plt.tight_layout(); plt.savefig("results/m2_pareto.png", dpi=110)


def m3():
    fl, lb, it = best_cfg()
    rows = []
    for s in SCENES:
        raw_gt, t_gt = read_frames(f"{DATA}/{s}_1000.bin")
        gt = gen_events(raw_first_L(raw_gt), t_gt)
        h, w = raw_gt.shape[1] // 2, raw_gt.shape[2] // 2
        for tag, noise in [("30", "off"), ("30n", "on")]:
            raw, t = read_frames(f"{DATA}/{s}_{tag}.bin")
            Li = isp_first_L(raw)
            conds = [("float raw-first", gen_events(raw_first_L(raw), t)),
                     ("float ISP-first", gen_events(Li, t)),
                     ("float ISP-first, C=0.25/2.4", gen_events(Li, t, C0 / 2.4)),
                     (f"fixed raw-first FL{fl}/LUT{lb}/i{it}", run_cpp(f"{DATA}/{s}_{tag}.bin", fl, lb, it, f"{RAWDIR}/m3.bin")[0])]
            for name, ev in conds:
                rows.append(dict(scene=s, noise=noise, path=name, **score(ev, gt, h, w, t)))
                r = rows[-1]
                print(f"M3 {s:8s} noise={noise:3s} {name:32s} F1={r['f1']:.4f} P={r['precision']:.4f} "
                      f"R={r['recall']:.4f} RMSE={r['count_rmse']:.3f} SSIM={r['ssim']:.4f} n={r['n_pred']}/{r['n_ref']}", flush=True)
    pd.DataFrame(rows).to_csv("results/m3_ablation.csv", index=False)


def cpu_model():
    if os.name == "nt":
        r = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True)
        return r.stdout.strip() or platform.processor()
    try:
        return next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return platform.processor()


def m4():
    print("CPU:", cpu_model())
    rows = []
    for fl, lb, it in sorted({best_cfg(), (16, 12, 1)}):
        for res in ["346x260", "1280x720"]:
            _, st = run_cpp(f"{DATA}/bench_{res}.bin", fl, lb, it)
            px = st["frames"] * st["W"] * st["H"]
            row = dict(res=res, fl=fl, lut_bits=lb, interp=it, frames=int(st["frames"]), events=int(st["events"]),
                       bin_ns_px=st["bin_s"] * 1e9 / px, log_ns_px=st["log_s"] * 1e9 / px, emit_ns_px=st["emit_s"] * 1e9 / px,
                       total_ns_px=st["ns_per_rawpx"], mev_s=st["mev_per_s"], fps=st["fps"], cpu=cpu_model())
            rows.append(row)
            print("M4 " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k != "cpu"))
    pd.DataFrame(rows).to_csv("results/m4_latency.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["bitwidth", "ablation", "latency"],
                    choices=["bitwidth", "ablation", "latency"])
    a = ap.parse_args()
    os.makedirs(RAWDIR, exist_ok=True)
    if not os.path.exists(EXE):
        sys.exit(f"missing {EXE}: build first")
    for st in a.stage:
        {"bitwidth": m2, "ablation": m3, "latency": m4}[st]()
