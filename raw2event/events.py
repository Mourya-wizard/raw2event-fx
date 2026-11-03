"""Float reference event generator: raw-first and ISP-first log-intensity paths.

Events: numpy structured array EV_F = (x u2, y u2, t f8 [us], p i1) at half resolution (2x2 binned).
"""
import numpy as np
import cv2
from .scenes import read_frames, BLACK, WHITE, SCENES, DATA

C0 = 0.25
EV_F = np.dtype([("x", "<u2"), ("y", "<u2"), ("t", "<f8"), ("p", "i1")])


def bin2x2(a):
    return a[..., 0::2, 0::2] + a[..., 0::2, 1::2] + a[..., 1::2, 0::2] + a[..., 1::2, 1::2]


def raw_first_L(raw, black=BLACK):
    """(N,H,W) raw -> (N,H/2,W/2) ln of linear luma Y=(R+G1+G2+B)/4 in DN. eps=0.25 DN (= cell sum 1)."""
    s = bin2x2(np.clip(raw.astype(np.float64) - black, 0, None))
    return np.log(np.maximum(s / 4.0, 0.25))


K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4


def isp_first_L(raw, black=BLACK, white=WHITE):
    """Bilinear demosaic -> gray-world WB (gains from frame 0) -> sRGB gamma -> Rec.709 luma -> 2x2 avg -> ln."""
    h, w = raw.shape[1:]
    m = np.zeros((3, h, w), np.float32)
    m[0, 0::2, 0::2] = 1; m[2, 1::2, 1::2] = 1; m[1] = 1 - m[0] - m[2]
    out, gains = [], None
    for fr in raw:
        x = np.clip(fr.astype(np.float32) - black, 0, None) / (white - black)
        rgb = np.stack([cv2.filter2D(x * m[c], -1, K_G if c == 1 else K_RB, borderType=cv2.BORDER_REFLECT_101)
                        for c in range(3)], -1)
        if gains is None:
            mu = rgb.reshape(-1, 3).mean(0)
            gains = mu[1] / np.maximum(mu, 1e-9)
        rgb = np.clip(rgb * gains, 0, 1)
        g = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
        y = g @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        out.append(bin2x2(y) / 4.0)
    return np.log(np.maximum(np.stack(out).astype(np.float64), 1e-3))


def gen_events(L, t_us, C=C0):
    """Deterministic frame-based generator, linear timestamp interpolation. L: (N,h,w) float."""
    w = L.shape[2]
    Lref = L[0].copy()
    out = []
    for k in range(1, len(L)):
        dL = L[k] - Lref
        a = np.abs(dL)
        n = np.floor(a / C).astype(np.int64)
        idx = np.flatnonzero(n)
        if idx.size:
            cnt = n.ravel()[idx]
            rep = np.repeat(idx, cnt)
            j = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + 1
            e = np.empty(rep.size, EV_F)
            e["x"], e["y"] = rep % w, rep // w
            e["t"] = t_us[k - 1] + (t_us[k] - t_us[k - 1]) * j * C / a.ravel()[rep]
            e["p"] = np.sign(dL.ravel()[rep])
            out.append(e)
        Lref += np.sign(dL) * n * C
    return np.concatenate(out) if out else np.empty(0, EV_F)


def events_for(scene, tag, path="raw"):
    raw, t = read_frames(f"{DATA}/{scene}_{tag}.bin")
    L = raw_first_L(raw) if path == "raw" else isp_first_L(raw)
    return gen_events(L, t), raw, t


def accum(ev, h, w, t0, t1):
    img = np.zeros((h, w))
    s = ev[(ev["t"] >= t0) & (ev["t"] < t1)]
    np.add.at(img, (s["y"], s["x"]), s["p"])
    return img
