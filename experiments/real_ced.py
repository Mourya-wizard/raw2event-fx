"""Experiment 6 (M6): validate our generators against REAL Color-DAVIS346 events (CED calibration bags).
APS image_raw (mono8 RGGB-type Bayer, 346x260) and DVS events come from the same pixels -> no registration needed.
Real events are 2x2-binned to our 173x130 cells. C is calibrated per path on ONE held-out bag (event count match), then frozen.
Metrics are computed only on cells never clipped (0 or 255) in any frame. Writes results/m6_ced.csv, m6_summary.csv, m6_ced.png.
Usage: python -m experiments.real_ced
"""
import os, sys, glob
for k, v in {"XDG_CACHE_HOME": "./.cache", "MPLCONFIGDIR": "./.cache/mpl",
             "HF_HOME": "./.cache/huggingface", "TORCH_HOME": "./.cache/torch"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
from raw2event.rosbag1 import read_bag, parse_image, parse_events
from raw2event.events import raw_first_L, isp_first_L, gen_events, EV_F, accum
from raw2event.scenes import write_frames
from raw2event.native import run_cpp, RAWDIR, best_cfg
from raw2event import metrics as M

CAL_BAG = "calib_fluorescent_density_infrared"
W, H, w, h = 346, 260, 173, 130
MIN_VALID = 0.05  # bags with fewer never-clipped cells are reported per bag but left out of the summary


def load(path):
    msgs, _ = read_bag(path, ["/dvs/image_raw", "/dvs/events"])
    ims = [parse_image(d) for d in msgs["/dvs/image_raw"]]
    ims = [x for x in ims if x[2].shape == (H, W)]  # a few truncated frames exist (e.g. 258 rows)
    t_us = np.round((np.array([x[0] for x in ims]) - ims[0][0]) * 1e6).astype(np.int64)
    fr = np.stack([x[2] for x in ims]).astype(np.uint16)
    e = np.concatenate([parse_events(d) for d in msgs["/dvs/events"]])
    t = (e["sec"].astype(np.float64) - np.floor(ims[0][0])) * 1e6 + e["nsec"] / 1e3 - (ims[0][0] % 1) * 1e6
    keep = (t > 0) & (t <= t_us[-1])
    real = np.empty(keep.sum(), EV_F)
    real["x"], real["y"], real["t"] = e["x"][keep] // 2, e["y"][keep] // 2, t[keep]
    real["p"] = np.where(e["p"][keep] > 0, 1, -1)
    clipped = ((fr == 0) | (fr == 255)).any(0)
    valid = ~(clipped[0::2, 0::2] | clipped[0::2, 1::2] | clipped[1::2, 0::2] | clipped[1::2, 1::2])
    return fr, t_us, real, valid


def only(ev, valid):
    return ev[valid[ev["y"], ev["x"]]]


def fixed_events(fr, t_us, C, cfg):
    write_frames(f"{RAWDIR}/m6_frames.bin", fr, t_us)
    return run_cpp(f"{RAWDIR}/m6_frames.bin", *cfg, f"{RAWDIR}/m6_ev.bin", extra=["--black", 0, "--c", C])[0]


def calibrate(make, n_target):
    """Bisection on log C so that the event count matches the real count (count is non-increasing in C)."""
    lo, hi = np.log(0.01), np.log(3.0)
    for _ in range(30):
        mid = (lo + hi) / 2
        if len(make(np.exp(mid))) > n_target: lo = mid
        else: hi = mid
    return float(np.exp((lo + hi) / 2))


def score(pred, real, t_us, valid):
    pred, real = only(pred, valid), only(real, valid)
    p, r, f1 = M.prf(pred, real, w, 0.5 * np.median(np.diff(t_us)))
    a, b = np.zeros(h * w), np.zeros(h * w)
    np.add.at(a, pred["y"].astype(int) * w + pred["x"], 1)
    np.add.at(b, real["y"].astype(int) * w + real["x"], 1)
    m = valid.ravel()
    return dict(n_pred=len(pred), n_real=len(real), count_ratio=len(pred) / max(len(real), 1), precision=p, recall=r, f1=f1,
                count_corr=float(np.corrcoef(a[m], b[m])[0, 1]), count_rmse=float(np.sqrt(np.mean((a[m] - b[m]) ** 2))),
                ssim=M.frames_ssim(pred, real, h, w, t_us))


if __name__ == "__main__":
    os.makedirs(RAWDIR, exist_ok=True)
    cfg = best_cfg()
    bags = sorted(glob.glob("data/ced/*.bag"))
    if not bags:
        sys.exit("no bags: download https://rpg.ifi.uzh.ch/datasets/CED/CED_calibration.zip and unzip into data/ced/")
    makers = {
        "float raw-first": lambda fr, t: (lambda C: gen_events(raw_first_L(fr, black=0), t, C)),
        "float ISP-first": lambda fr, t: (lambda C: gen_events(isp_first_L(fr, black=0, white=255), t, C)),
    }
    fr, t_us, real, valid = load(f"data/ced/{CAL_BAG}.bag")
    Cs = {k: calibrate(lambda C: only(mk(fr, t_us)(C), valid), len(only(real, valid))) for k, mk in makers.items()}
    print(f"calibrated on {CAL_BAG} (valid cells {valid.mean():.3f}): " + ", ".join(f"{k} C={v:.4f}" for k, v in Cs.items()))
    rows, example = [], None
    for b in bags:
        name = os.path.basename(b)[:-4]
        if name == CAL_BAG:
            continue
        fr, t_us, real, valid = load(b)
        preds = {k: mk(fr, t_us)(Cs[k]) for k, mk in makers.items()}
        preds[f"fixed raw-first FL{cfg[0]}/LUT{cfg[1]}/i{cfg[2]}"] = fixed_events(fr, t_us, Cs["float raw-first"], cfg)
        for k, ev in preds.items():
            rows.append(dict(bag=name, path=k, valid_frac=float(valid.mean()), C=Cs["float raw-first" if k.startswith("fixed") else k],
                             **score(ev, real, t_us, valid)))
            r = rows[-1]
            print(f"M6 {name:34s} {k:30s} F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} ratio={r['count_ratio']:.3f} "
                  f"corr={r['count_corr']:.4f} SSIM={r['ssim']:.4f} valid={r['valid_frac']:.3f}", flush=True)
        if name == "calib_low":
            example = (name, fr, t_us, real, valid, preds)
    df = pd.DataFrame(rows)
    df.to_csv("results/m6_ced.csv", index=False)
    print("excluded from summary (valid cells < 5%):", sorted(df[df.valid_frac < MIN_VALID].bag.unique()))
    df = df[df.valid_frac >= MIN_VALID]
    cols =["f1", "precision", "recall", "count_ratio", "count_corr", "count_rmse", "ssim"]
    s = df.groupby("path")[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.to_csv("results/m6_summary.csv")
    print(f"test bags: {df.bag.nunique()}")
    print(s.to_string(float_format=lambda v: f"{v:.4f}"))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name, fr, t_us, real, valid, preds = example
    k = len(t_us) // 2
    t0, t1 = t_us[k], t_us[k + 4]
    show = [("real DAVIS events (2x2 binned)", real)] + list(preds.items())
    fig, ax = plt.subplots(1, len(show) + 2, figsize=(4 * (len(show) + 2), 3.6))
    ax[0].imshow(fr[k], cmap="gray"); ax[0].set_title(f"{name}: APS image_raw")
    for i, (lbl, ev) in enumerate(show):
        a = accum(only(ev, valid), h, w, t0, t1); m = max(np.abs(a).max(), 1)
        ax[i + 1].imshow(a, cmap="bwr", vmin=-m, vmax=m); ax[i + 1].set_title(f"{lbl}\n{(t1 - t0) / 1e3:.0f} ms window", fontsize=8)
    for a_ in ax[:-1]: a_.axis("off")
    paths = list(s.index)
    ax[-1].bar(range(len(paths)), s.f1_mean, yerr=s.f1_std, capsize=3)
    ax[-1].set_xticks(range(len(paths))); ax[-1].set_xticklabels([p.replace(" ", "\n", 1) for p in paths], fontsize=7)
    ax[-1].set_ylabel(f"F1 vs real (mean±std, {df.bag.nunique()} bags)")
    plt.tight_layout(); plt.savefig("results/m6_ced.png", dpi=100)
    print("saved results/m6_ced.png")
