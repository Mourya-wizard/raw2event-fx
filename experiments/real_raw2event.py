"""Experiment 7 (M7): validate raw-first / ISP-first / fixed-point events against REAL DAVIS346 events (Raw2Event HF dataset).
Per prefix: Pi Camera 3 10-bit Bayer MKV (692x520, 60 fps) + DAVIS346 AEDAT4 (events + APS frames) + meta .dat.
Registration: AprilTag (tag36h11) corners in APS frames vs nearest Pi frame -> one RANSAC homography per prefix
(planar screen, rigid rig). Pi luma is resampled onto the DAVIS grid, so all generators emit DAVIS-coordinate events.
C (per path) and a global time lag are calibrated on CAL prefixes, frozen, and evaluated on the rest.
Usage: python -m experiments.real_raw2event [prefix_list.txt]   -> results/m7_raw2event.csv, results/m7_summary.csv, results/m7_raw2event.png
"""
import os, datetime, zoneinfo
for k, v in {"XDG_CACHE_HOME": "./.cache", "MPLCONFIGDIR": "./.cache/mpl",
             "HF_HOME": "./.cache/huggingface", "TORCH_HOME": "./.cache/torch"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
import cv2, av, aedat
from raw2event.scenes import write_frames
from raw2event.events import raw_first_L, isp_first_L, gen_events, bin2x2, EV_F
from raw2event.native import run_cpp, RAWDIR
from raw2event import metrics as M

D, W, H, BLACK = "data/raw2event", 346, 260, 64   # IMX708 10-bit black level 64 (measured min 66)
TOL = 0.5 * 1e6 / 60                               # half the 60 fps Pi frame period, us
TZ = zoneinfo.ZoneInfo("Europe/Brussels")          # meta wall-clock strings are local time (CET)
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv2.aruco.DetectorParameters())
FIX = (10, 8, 1)                                   # M2 best: FL, LUT_BITS, interp


def load(p):
    raw = np.stack([f.to_ndarray() for f in av.open(f"{D}/{p}.mkv").decode(video=0)])
    b = open(f"{D}/{p}.dat", "rb").read()           # 38-byte records: f64 rel-time us, 26-char wall time, 4 pad
    assert len(b) % 38 == 0 and len(b) // 38 == len(raw), (len(b), len(raw))
    wall = [b[i + 8:i + 34].decode() for i in range(0, len(b), 38)]
    wall = np.array([datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=TZ).timestamp() * 1e6 for s in wall])
    rel = np.frombuffer(b"".join(b[i:i + 8] for i in range(0, len(b), 38)), "<f8")
    t = rel + np.median(wall - rel)  # sensor clock (step std ~1 us) anchored to wall clock (step std 0.2-0.6 ms)
    ev, aps = [], []
    for pkt in aedat.Decoder(f"{D}/{p}.aedat4"):
        if "events" in pkt: ev.append(pkt["events"])
        if "frame" in pkt: aps.append((pkt["frame"]["t"], np.squeeze(pkt["frame"]["pixels"])))
    ev = np.concatenate(ev)
    return raw, np.round(t).astype(np.int64), ev, aps


def to8(a):
    return cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def tag_corners(img8, up=1):
    c, ids, _ = DET.detectMarkers(cv2.resize(img8, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else img8)
    return None if ids is None or len(ids) != 1 else c[0].reshape(4, 2) / up


def homography(Y, t, aps):
    """Pi-binned (H,W) coords -> DAVIS coords, from tag corners in time-paired frames."""
    src, dst = [], []
    for ta, a in aps:
        k = int(np.argmin(np.abs(t - ta)))
        if abs(t[k] - ta) > TOL: continue
        cd, cp = tag_corners(to8(a), 3), tag_corners(to8(Y[k]))
        if cd is not None and cp is not None:
            src.append(cp); dst.append(cd)
    if len(src) < 3: return None, len(src), np.nan
    src, dst = np.concatenate(src), np.concatenate(dst)
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    res = np.linalg.norm(cv2.perspectiveTransform(src[None], Hm)[0] - dst, axis=1)[inl.ravel() > 0]
    return Hm, len(src) // 4, float(np.sqrt(np.mean(res ** 2)))


def warp(img_stack, Hm):
    s = np.sqrt(abs(np.linalg.det(Hm[:2, :2])))     # DAVIS px per Pi px (<1): anti-alias blur before resampling
    sig = max(0.5 / s - 0.5, 0)
    return np.stack([cv2.warpPerspective(cv2.GaussianBlur(f, (0, 0), sig) if sig else f, Hm, (W, H), flags=cv2.INTER_LINEAR)
                     for f in img_stack.astype(np.float32)])


def prepare(p):
    raw, t, ev, aps = load(p)
    Y = bin2x2(np.clip(raw.astype(np.float32) - BLACK, 0, None)) / 4          # linear luma, Pi grid
    Hm, npair, rms = homography(Y, t, aps)
    if Hm is None: return None
    mask = cv2.erode((warp(np.ones((1, H, W)), Hm)[0] > 0.999).astype(np.uint8), np.ones((7, 7)))
    Yw = warp(Y, Hm)
    q = np.round(4 * Yw).astype(np.int64)                                     # synthetic raw: cell sum = round(4*Y)
    sraw = np.zeros((len(q), 2 * H, 2 * W), np.uint16)
    for i, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        sraw[:, dy::2, dx::2] = BLACK + q // 4 + (q % 4 > i)
    Li = np.log(np.maximum(warp(np.exp(isp_first_L(raw, BLACK, 1023)), Hm), 1e-3))  # ISP on true Bayer, then resample
    t0 = t[0]
    ys, xs = np.nonzero(mask)
    ev = ev[(ev["t"] >= max(t0, ev["t"][0])) & (ev["t"] <= t[-1]) & (mask[ev["y"], ev["x"]] > 0)]
    real = np.empty(len(ev), EV_F)
    real["x"], real["y"], real["t"], real["p"] = ev["x"], ev["y"], ev["t"] - t0, np.where(ev["on"] if "on" in ev.dtype.names else ev["p"], 1, -1)
    return dict(p=p, t=t - t0, sraw=sraw, Lr=raw_first_L(sraw, BLACK), Li=Li, real=real, mask=mask, rms=rms, npair=npair,
                bbox=(ys.min(), ys.max() + 1, xs.min(), xs.max() + 1), Y=Y, Hm=Hm)


def clip_win(e, d, lag=0.0):
    e = e.copy(); e["t"] = e["t"] + lag
    lo, hi = d["real"]["t"].min(), d["real"]["t"].max()
    return e[(e["t"] >= lo) & (e["t"] <= hi) & (d["mask"][e["y"], e["x"]] > 0)]


def lag_of(pred, real):
    """Lag (us) to ADD to pred times maximising cross-correlation of 2 ms event-rate histograms (+-300 ms)."""
    bins = np.arange(real["t"].min() - 4e5, real["t"].max() + 4e5, 2000)
    a, b = np.histogram(pred["t"], bins)[0].astype(float), np.histogram(real["t"], bins)[0].astype(float)
    a, b = a - a.mean(), b - b.mean()
    lags = np.arange(-150, 151)
    return 2000.0 * lags[np.argmax([np.dot(np.roll(a, l), b) for l in lags])]


def events(d, path, C, lag):
    if path == "float raw-first": e = gen_events(d["Lr"], d["t"], C)
    elif path == "float ISP-first": e = gen_events(d["Li"], d["t"], C)
    else:
        write_frames(f"{RAWDIR}/m7_frames.bin", d["sraw"], d["t"])
        ei = run_cpp(f"{RAWDIR}/m7_frames.bin", *FIX, f"{RAWDIR}/m7_ev.bin", extra=("--c", C, "--black", BLACK))[0]
        e = np.empty(len(ei), EV_F)
        for k in "xytp": e[k] = ei[k]
    return clip_win(e, d, lag)


def calib_C(cal, L_key):
    """Bisection on log C so total predicted events == total real events over CAL prefixes (within mask/window)."""
    target, lo, hi = sum(len(d["real"]) for d in cal), np.log(0.01), np.log(2.0)
    for _ in range(14):
        c = np.exp((lo + hi) / 2)
        n = sum(len(clip_win(gen_events(d[L_key], d["t"], c), d)) for d in cal)
        lo, hi = (np.log(c), hi) if n > target else (lo, np.log(c))
    return float(np.exp((lo + hi) / 2))


def match_mask(ka, ta, kb, tb, tol):
    """Two-pointer one-to-one match on (key, time); returns matched masks. Same algorithm as metrics.match_count."""
    ia, ib = np.lexsort((ta, ka)), np.lexsort((tb, kb))
    A, TA, B, TB = ka[ia].tolist(), ta[ia].tolist(), kb[ib].tolist(), tb[ib].tolist()
    ma, mb, i, j = np.zeros(len(A), bool), np.zeros(len(B), bool), 0, 0
    while i < len(A) and j < len(B):
        if A[i] != B[j]:
            if A[i] < B[j]: i += 1
            else: j += 1
        elif abs(TA[i] - TB[j]) <= tol: ma[i] = mb[j] = True; i += 1; j += 1
        elif TA[i] < TB[j]: i += 1
        else: j += 1
    out_a, out_b = np.zeros(len(A), bool), np.zeros(len(B), bool)
    out_a[ia], out_b[ib] = ma, mb
    return out_a, out_b


def match_radius(pred, real, w, h, tol, r):
    """Greedy one-to-one match, same polarity, |dt|<=tol, spatial offset |d|<=r (nearer offsets tried first)."""
    offs = sorted([(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r], key=lambda o: o[0] ** 2 + o[1] ** 2)
    px, py, pp, pt = pred["x"].astype(np.int64), pred["y"].astype(np.int64), pred["p"] > 0, pred["t"].astype(np.float64)
    kb_all = (real["y"].astype(np.int64) * w + real["x"]) * 2 + (real["p"] > 0)
    up, ur, tp = np.ones(len(pred), bool), np.ones(len(real), bool), 0
    for dx, dy in offs:
        ia, ib = np.flatnonzero(up), np.flatnonzero(ur)
        x, y = px[ia] + dx, py[ia] + dy
        ka = np.where((x >= 0) & (x < w) & (y >= 0) & (y < h), (y * w + x) * 2 + pp[ia], -1 - np.arange(len(ia)))
        ma, mb = match_mask(ka, pt[ia], kb_all[ib], real["t"][ib].astype(np.float64), tol)
        up[ia[ma]] = False; ur[ib[mb]] = False
        tp += int(ma.sum())
    p, rc = tp / max(len(pred), 1), tp / max(len(real), 1)
    return 2 * p * rc / max(p + rc, 1e-12)


def score(e, d):
    real, (y0, y1, x0, x1) = d["real"], d["bbox"]
    h, w = y1 - y0, x1 - x0
    a, b = e.copy(), real.copy()
    for z in (a, b): z["x"] -= np.uint16(x0); z["y"] -= np.uint16(y0)
    ca, cb = np.zeros(h * w), np.zeros(h * w)
    np.add.at(ca, a["y"].astype(int) * w + a["x"], 1); np.add.at(cb, b["y"].astype(int) * w + b["x"], 1)
    m = d["mask"][y0:y1, x0:x1].ravel() > 0
    edges = np.arange(b["t"].min(), b["t"].max(), 2e6 / 60)                 # 33 ms windows
    out = dict(n_pred=len(a), n_real=len(b), count_ratio=len(a) / max(len(b), 1),
               count_r=float(np.corrcoef(ca[m], cb[m])[0, 1]), count_rmse=float(np.sqrt(np.mean((ca[m] - cb[m]) ** 2))),
               ssim=M.frames_ssim(a, b, h, w, edges))
    for r in (0, 1, 2):
        out[f"f1_r{r}"] = match_radius(a, b, w, h, TOL, r)
    return out


if __name__ == "__main__":
    import time
    T0 = time.time()
    _r = np.array([(5, 5, 100.0, 1), (9, 9, 100.0, -1)], EV_F)
    _p = np.array([(6, 5, 150.0, 1), (9, 9, 100.0, 1)], EV_F)       # 1 px off; wrong polarity
    assert match_radius(_p, _r, 20, 20, 60, 0) == 0 and abs(match_radius(_p, _r, 20, 20, 60, 1) - 0.5) < 1e-9
    os.makedirs(RAWDIR, exist_ok=True)
    import sys
    pfx = [l.strip() for l in open(sys.argv[1] if len(sys.argv) > 1 else f"{D}/prefixes.txt") if l.strip()]
    cal_names = set(pfx[::4])                                                # 5 calibration prefixes, 15 test
    data, skipped = {}, []
    for p in pfx:
        d = prepare(p)
        if d is None: skipped.append(p); print("SKIP (no homography)", p); continue
        data[p] = d
        print(f"prep {p}: tag pairs={d['npair']} H-rms={d['rms']:.3f}px real={len(d['real'])} ({time.time()-T0:.0f}s)", flush=True)
    cal = [data[p] for p in pfx if p in cal_names and p in data]
    C = {"float raw-first": calib_C(cal, "Lr"), "float ISP-first": calib_C(cal, "Li")}
    C["fixed raw-first FL10/LUT8/i1"] = C["float raw-first"]
    lag = float(np.median([lag_of(clip_win(gen_events(d["Lr"], d["t"], C["float raw-first"]), d), d["real"]) for d in cal]))
    print("calibrated C:", C, "global lag us:", lag, flush=True)
    rows = []
    for p, d in data.items():
        resid = lag_of(events(d, "float raw-first", C["float raw-first"], lag), d["real"])
        for path, c in C.items():
            rows.append(dict(prefix=p, split="cal" if p in cal_names else "test", path=path, C=c, lag_us=lag, resid_lag_us=resid,
                             h_rms_px=d["rms"], tag_pairs=d["npair"], **score(events(d, path, c, lag), d)))
            r = rows[-1]
            print(f"M7 {p[:28]:28s} {r['split']:4s} {path:28s} F1 r0/1/2={r['f1_r0']:.3f}/{r['f1_r1']:.3f}/{r['f1_r2']:.3f} "
                  f"ratio={r['count_ratio']:.3f} r={r['count_r']:.3f} ssim={r['ssim']:.3f} resid={resid/1e3:.0f}ms", flush=True)
    df = pd.DataFrame(rows); df.to_csv("results/m7_raw2event.csv", index=False)
    mets = ["f1_r0", "f1_r1", "f1_r2", "count_ratio", "count_r", "count_rmse", "ssim"]
    summ = df[df.split == "test"].groupby("path")[mets].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["n_test"] = df[df.split == "test"].groupby("path").size()
    summ.to_csv("results/m7_summary.csv"); print(summ.T.to_string(float_format=lambda v: f"{v:.4f}"))
    print("skipped:", skipped, f"runtime {time.time()-T0:.0f}s")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = data[next(p for p in pfx if p in data and p not in cal_names)]
    (y0, y1, x0, x1), tm = d["bbox"], np.median(d["real"]["t"])
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    ax[0, 0].imshow(d["Y"][np.argmin(np.abs(d["t"] - tm))] ** (1 / 2.2), cmap="gray"); ax[0, 0].set_title("Pi raw (binned, Pi grid)")
    for j, (name, e) in enumerate([("real DAVIS346", d["real"]), ("float raw-first", events(d, "float raw-first", C["float raw-first"], lag)),
                                    ("float ISP-first", events(d, "float ISP-first", C["float ISP-first"], lag))]):
        s = e[(e["t"] >= tm) & (e["t"] < tm + 50_000)]
        img = np.zeros((H, W)); np.add.at(img, (s["y"], s["x"]), s["p"])
        v = max(np.percentile(np.abs(img[img != 0]), 99) if (img != 0).any() else 1, 1)  # hot pixels would dominate max
        ax[0, j + 1].imshow(img[y0:y1, x0:x1], cmap="bwr", vmin=-v, vmax=v); ax[0, j + 1].set_title(f"{name}, 50 ms (DAVIS coords)")
    for a_ in ax[0]: a_.axis("off")
    te = df[df.split == "test"]
    for j, (grp, title) in enumerate([(["f1_r0", "f1_r1", "f1_r2"], "F1 vs real (radius 0/1/2 px)"), (["count_r", "ssim"], "count-map Pearson r / SSIM"),
                                       (["count_ratio"], "event count ratio pred/real")]):
        g = te.groupby("path")[grp]
        g.mean().T.plot.bar(ax=ax[1, j], yerr=g.std().T, capsize=3, rot=0); ax[1, j].set_title(f"{title} (test, n={te.prefix.nunique()})")
        ax[1, j].legend(fontsize=7)
    ax[1, 3].axis("off")
    ax[1, 3].text(0, 0.5, "\n".join(f"{k}: C={v:.4f}" for k, v in C.items()) + f"\nlag={lag/1e3:.1f} ms\nH-rms med={np.median([d['rms'] for d in data.values()]):.2f}px", fontsize=9)
    plt.tight_layout(); plt.savefig("results/m7_raw2event.png", dpi=100)
    print("saved results/m7_raw2event.png")
