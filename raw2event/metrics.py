"""Event-stream metrics. Events: structured arrays with fields x, y, t (us), p (+1/-1)."""
import numpy as np
from skimage.metrics import structural_similarity


def match_count(pred, ref, w, tol_us):
    """One-to-one matches: same pixel (radius 0), same polarity, |dt| <= tol_us.
    Per (pixel, polarity) both lists are time-sorted; greedy two-pointer matching is a maximum matching in 1-D."""
    ka = (pred["y"].astype(np.int64) * w + pred["x"]) * 2 + (pred["p"] > 0)
    kb = (ref["y"].astype(np.int64) * w + ref["x"]) * 2 + (ref["p"] > 0)
    ia, ib = np.lexsort((pred["t"], ka)), np.lexsort((ref["t"], kb))
    ka, ta = ka[ia].tolist(), pred["t"][ia].astype(np.float64).tolist()
    kb, tb = kb[ib].tolist(), ref["t"][ib].astype(np.float64).tolist()
    i = j = tp = 0
    while i < len(ka) and j < len(kb):
        if ka[i] != kb[j]:
            if ka[i] < kb[j]: i += 1
            else: j += 1
        elif abs(ta[i] - tb[j]) <= tol_us:
            tp += 1; i += 1; j += 1
        elif ta[i] < tb[j]: i += 1
        else: j += 1
    return tp


def prf(pred, ref, w, tol_us):
    tp = match_count(pred, ref, w, tol_us)
    p = tp / max(len(pred), 1)
    r = tp / max(len(ref), 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def count_rmse(pred, ref, h, w):
    """RMSE over pixels of total event counts (both polarities)."""
    a, b = np.zeros(h * w), np.zeros(h * w)
    np.add.at(a, pred["y"].astype(np.int64) * w + pred["x"], 1)
    np.add.at(b, ref["y"].astype(np.int64) * w + ref["x"], 1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def frames_ssim(pred, ref, h, w, t_edges):
    """Mean SSIM of signed polarity-accumulated frames over consecutive [t_edges[i], t_edges[i+1]) windows."""
    out = []
    for t0, t1 in zip(t_edges[:-1], t_edges[1:]):
        fa, fb = np.zeros((h, w)), np.zeros((h, w))
        for e, f in ((pred, fa), (ref, fb)):
            s = e[(e["t"] > t0) & (e["t"] <= t1)]
            np.add.at(f, (s["y"], s["x"]), s["p"])
        rng = 2 * max(np.abs(fa).max(), np.abs(fb).max(), 1)
        out.append(structural_similarity(fa, fb, data_range=rng))
    return float(np.mean(out))
