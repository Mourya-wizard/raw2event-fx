"""Experiment 5 (M5): task-level validation. MNIST digits -> linear radiance + 3-saccade motion -> 10-bit raw -> events
(float raw-first, fixed raw-first at several FL, float ISP-first) -> tiny CNN. Writes results/m5_task.csv, results/m5_task.png.
Samples are stacked vertically into one tall image (H = B*68) so the per-pixel generators run in one call.
Usage: python -m experiments.mnist_task
"""
import os, sys, time
for k, v in {"XDG_CACHE_HOME": "./.cache", "MPLCONFIGDIR": "./.cache/mpl",
             "HF_HOME": "./.cache/huggingface", "TORCH_HOME": "./.cache/torch"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
import cv2
import torch, torch.nn as nn
from raw2event.scenes import mosaic, to_raw, write_frames, DATA
from raw2event.events import raw_first_L, isp_first_L, gen_events
from raw2event.native import run_cpp, RAWDIR, best_cfg
from raw2event import metrics as M

N_TR, N_TE, FR, FPS, HW, TB = 4000, 1000, 40, 100, 34, 4   # samples, frames, fps, half-res side, time bins
SAC = np.array([[0, 0], [3, 3], [6, 0], [0, 0]], np.float32)  # half-res px; 3 saccades (N-MNIST style triangle)


def load_digits():
    try:
        from torchvision.datasets import MNIST
        ds = MNIST(f"{DATA}/mnist", train=True, download=True)
        x, y = ds.data.numpy()[: N_TR + N_TE], ds.targets.numpy()[: N_TR + N_TE]
        return x.astype(np.float32) / 255, y, "MNIST"
    except Exception as e:  # fallback: 10 synthetic shape classes
        print("MNIST unavailable, using synthetic shapes:", e)
        rng, xs, ys = np.random.default_rng(0), [], []
        for i in range(N_TR + N_TE):
            c, im = i % 10, np.zeros((28, 28), np.float32)
            cx, cy, r = rng.uniform(11, 17, 2).tolist() + [int(rng.integers(6, 10))]
            p = (int(cx), int(cy))
            if c < 5:
                cv2.circle(im, p, r, 1, 2 if c % 2 == 0 else -1)  # 0-4: circle outline (even) / filled (odd)
            else:
                a = (c - 5) * np.pi / 5
                d = (int(r * np.cos(a)), int(r * np.sin(a)))
                cv2.line(im, (p[0] - d[0], p[1] - d[1]), (p[0] + d[0], p[1] + d[1]), 1, 3)
            if c in (2, 3): cv2.rectangle(im, (p[0] - 3, p[1] - 3), (p[0] + 3, p[1] + 3), 1, -1 if c == 2 else 1)
            xs.append(im); ys.append(c)
        return np.stack(xs), np.array(ys), "synthetic-shapes"


def render(digits, seed):
    """digits (B,28,28) -> raw (FR, B*68, 68) uint16; random background per sample, log-uniform [0.01, 0.33]."""
    rng = np.random.default_rng(seed)
    bg = np.exp(rng.uniform(np.log(0.01), np.log(0.33), len(digits))).astype(np.float32)
    big = [cv2.resize(d, (56, 56), interpolation=cv2.INTER_LINEAR) for d in digits]  # full-res Bayer px
    out = np.empty((FR, len(digits) * 68, 68), np.uint16)
    for k in range(FR):
        s = k / (FR - 1) * 3                     # position along the 3 saccades
        i = min(int(s), 2)
        off = (SAC[i] + (s - i) * (SAC[i + 1] - SAC[i])) * 2 + 3  # full-res px, sub-pixel
        Mw = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
        for b, d in enumerate(big):
            dd = cv2.warpAffine(d, Mw, (68, 68), flags=cv2.INTER_LINEAR)
            lin = bg[b] * (1 + 2 * dd)
            out[k, b * 68:(b + 1) * 68] = to_raw(mosaic(np.repeat(lin[..., None], 3, 2)))
    return out


def to_tensor(ev, B):
    """events (stacked coords) -> (B, 2*TB, 34, 34) float32 counts, channel = pol*TB + time bin."""
    x = np.zeros((B, 2, TB, HW, HW), np.float32)
    tb = np.clip((ev["t"] / (1e6 * (FR - 1) / FPS) * TB).astype(int), 0, TB - 1)
    np.add.at(x, (ev["y"] // HW, (ev["p"] > 0).astype(int), tb, ev["y"] % HW, ev["x"]), 1)
    return x.reshape(B, 2 * TB, HW, HW)


def net():
    return nn.Sequential(nn.Conv2d(2 * TB, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


def train(xtr, ytr, seed, dev):
    torch.manual_seed(seed)
    m, g = net().to(dev), torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    X, Y = torch.log1p(torch.from_numpy(xtr)).to(dev), torch.from_numpy(ytr).long().to(dev)
    for _ in range(10):
        for idx in torch.randperm(len(X), generator=g).split(64):
            opt.zero_grad()
            nn.functional.cross_entropy(m(X[idx]), Y[idx]).backward()
            opt.step()
    return m


@torch.no_grad()
def acc(m, x, y, dev):
    m.eval()
    return (m(torch.log1p(torch.from_numpy(x)).to(dev)).argmax(1).cpu().numpy() == y).mean()


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RAWDIR, exist_ok=True)
    digits, labels, src = load_digits()
    bfl, blb, bit = best_cfg()
    fls = [bfl] + [bfl - 2 * i for i in (1, 2, 3)]  # best + 3 lower FL (same LUT/interp)
    conds = ["float raw-first", "float ISP-first"] + [f"fixed FL{f}" for f in fls]
    X = {c: [] for c in conds}
    f1 = {f: [] for f in fls}
    t0 = time.time()
    for start, n in [(0, N_TR), (N_TR, N_TE)]:  # train / test splits, chunks of 1000 samples
        for c0 in range(start, start + n, 1000):
            raw = render(digits[c0:c0 + 1000], seed=c0)
            t_us = np.round(np.arange(FR) * 1e6 / FPS).astype(np.int64)
            B = raw.shape[1] // 68
            evf = gen_events(raw_first_L(raw), t_us)
            X["float raw-first"].append(to_tensor(evf, B))
            X["float ISP-first"].append(to_tensor(gen_events(isp_first_L(raw), t_us), B))
            write_frames(f"{RAWDIR}/m5_frames.bin", raw, t_us)
            for f in fls:
                ev, _ = run_cpp(f"{RAWDIR}/m5_frames.bin", f, blb, bit, f"{RAWDIR}/m5_ev.bin")
                X[f"fixed FL{f}"].append(to_tensor(ev, B))
                if start == N_TR:
                    f1[f].append(M.prf(ev, evf, HW, 0.5 * 1e6 / FPS)[2])
            print(f"rendered+events samples {c0}..{c0 + B} ({time.time() - t0:.0f}s)", flush=True)
    X = {c: np.concatenate(v) for c, v in X.items()}
    ytr, yte = labels[:N_TR], labels[N_TR:]
    rows = []
    for seed in range(3):
        models = {c: train(X[c][:N_TR], ytr, seed, dev) for c in conds}
        for c in conds:
            rows.append(dict(seed=seed, train=c, test=c, acc=acc(models[c], X[c][N_TR:], yte, dev)))
            if c.startswith("fixed"):
                rows.append(dict(seed=seed, train="float raw-first", test=c,
                                 acc=acc(models["float raw-first"], X[c][N_TR:], yte, dev)))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/m5_task_runs.csv", index=False)
    s = df.groupby(["train", "test"], as_index=False).acc.agg(["mean", "std"]).reset_index(drop=True)
    s["mnist_event_f1_vs_float"] = [np.mean(f1[int(t[8:])]) if t.startswith("fixed") else np.nan for t in s.test]
    s.to_csv("results/m5_task.csv", index=False)
    print(f"data={src} train={N_TR} test={N_TE} LUT={blb} interp={bit}")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m2 = pd.read_csv("results/m2_summary.csv")
    m2 = m2[(m2.lut_bits == blb) & (m2.interp == bit)].sort_values("fl")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    same = s[(s.train == s.test) & s.test.str.startswith("fixed")].assign(fl=lambda d: d.test.str[8:].astype(int)).sort_values("fl")
    cross = s[(s.train != s.test)].assign(fl=lambda d: d.test.str[8:].astype(int)).sort_values("fl")
    ax[0].errorbar(same.fl, same["mean"], same["std"], marker="o", capsize=3, label="fixed raw-first (train=test)")
    ax[0].errorbar(cross.fl, cross["mean"], cross["std"], marker="s", capsize=3, label="train float raw, test fixed")
    for c, ls in [("float raw-first", "--"), ("float ISP-first", ":")]:
        r = s[(s.train == c) & (s.test == c)].iloc[0]
        ax[0].axhline(r["mean"], ls=ls, c="k", label=f"{c} {r['mean']:.3f}±{r['std']:.3f}")
    ax[0].set_xlabel("FL bits"); ax[0].set_ylabel(f"test accuracy ({src}, 3 seeds)"); ax[0].legend(fontsize=7)
    ax[1].plot(m2.fl, m2.f1_mean, "o-", label="M2 scenes: F1 vs float (mean)")
    ax[1].plot(same.fl, same.mnist_event_f1_vs_float, "s-", label=f"{src} test set: F1 vs float")
    ax[1].axhline(0.99, ls="--", c="gray", lw=0.8)
    ax[1].set_xlabel("FL bits"); ax[1].set_ylabel(f"event F1 (LUT{blb}, interp={bit})"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig("results/m5_task.png", dpi=110)
    print("saved results/m5_task.png, results/m5_task.csv")
