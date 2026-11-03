"""Synthetic LINEAR-radiance scenes -> RGGB 10-bit raw frames (frames.bin format).

frames.bin layout (little-endian): uint32 N, uint32 W, uint32 H, int32 t_us[N], uint16 raw[N][H][W].
Raw: RGGB (R at (0,0)), black level 64, white 1023, noise optional (Poisson-Gaussian).
CLI (writes data/*.bin): python -m raw2event.scenes [--dur 0.5] [--video clip.mp4]
"""
import os, sys, argparse
for k, v in {"XDG_CACHE_HOME": "./.cache", "MPLCONFIGDIR": "./.cache/mpl",
             "HF_HOME": "./.cache/huggingface", "TORCH_HOME": "./.cache/torch"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import cv2
from skimage import data as skdata

W, H, BLACK, WHITE = 346, 260, 64, 1023
SCENES = ["texture", "checker", "bars"]
DATA = "data"


def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


_TEX = None
def _texture():
    global _TEX
    if _TEX is None:
        img = skdata.astronaut().astype(np.float32) / 255.0
        img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_CUBIC)
        _TEX = np.clip(srgb_to_linear(img), 0, 1).astype(np.float32) * 0.9 + 0.01
    return _TEX


def radiance(scene, t, w=W, h=H):
    """Linear RGB radiance (h, w, 3) float32 in (0, 1] at time t seconds."""
    if scene == "texture":  # sub-pixel translation of a linearised photo
        dx, dy = 20 + 60.0 * t, 20 + 25.0 * t
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        tex = _texture()
        if w > tex.shape[1] - 200 or h > tex.shape[0] - 200:  # bench sizes: upscale the texture
            s = max((w + 200) / tex.shape[1], (h + 200) / tex.shape[0])
            M[:, :2] *= s  # forward map dst = s*src - d  ->  dst(x) = tex((x + d) / s)
        return cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if scene == "checker":  # rotating soft checkerboard
        a = 0.8 * t
        u = (xx - w / 2) * np.cos(a) + (yy - h / 2) * np.sin(a)
        v = -(xx - w / 2) * np.sin(a) + (yy - h / 2) * np.cos(a)
        c = 0.5 + 0.5 * np.tanh(3 * np.sin(np.pi * u / 32) * np.sin(np.pi * v / 32))
        r = 0.05 + 0.6 * c
    elif scene == "bars":  # moving bars over dark (left) + bright (right) background
        bg = 0.03 + (0.30 - 0.03) / (1 + np.exp(-(xx - w / 2) / 4))
        bar = 0.5 + 0.5 * np.tanh(4 * np.sin(2 * np.pi * (xx - 90.0 * t) / 80))
        r = bg * (1 + 2.0 * bar)
    else:
        raise ValueError(scene)
    return np.repeat(r[..., None], 3, axis=2).astype(np.float32)


def mosaic(rgb):
    """RGGB mosaic of (h, w, 3) -> (h, w)."""
    out = rgb[..., 1].copy()
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return out


def to_raw(lin, rng=None, full_well=4000.0, read_noise=2.0):
    """Linear [0,1] mosaic -> 10-bit raw uint16. rng=None disables noise."""
    dn = lin * (WHITE - BLACK)
    if rng is not None:
        e = rng.poisson(np.clip(lin, 0, None) * full_well)
        dn = e * (WHITE - BLACK) / full_well + rng.normal(0, read_noise, lin.shape)
    return np.clip(np.round(BLACK + dn), 0, WHITE).astype(np.uint16)


def write_frames(path, frames, t_us):
    frames = np.ascontiguousarray(frames, dtype="<u2")
    n, h, w = frames.shape
    with open(path, "wb") as f:
        np.array([n, w, h], "<u4").tofile(f)
        np.asarray(t_us, "<i4").tofile(f)
        frames.tofile(f)


def read_frames(path):
    with open(path, "rb") as f:
        n, w, h = np.fromfile(f, "<u4", 3)
        t = np.fromfile(f, "<i4", n).astype(np.int64)
        fr = np.fromfile(f, "<u2", n * h * w).reshape(n, h, w)
    return fr, t


def render(scene, fps, dur, noise=False, w=W, h=H, seed=0):
    n = int(round(dur * fps)) + 1
    t_us = np.round(np.arange(n) * 1e6 / fps).astype(np.int64)
    rng = np.random.default_rng(seed) if noise else None
    return np.stack([to_raw(mosaic(radiance(scene, t / 1e6, w, h)), rng) for t in t_us]), t_us


def video_to_raw(path, max_frames=300, seed=0):
    cap, rng, out = cv2.VideoCapture(path), np.random.default_rng(seed), []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while len(out) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = srgb_to_linear(bgr[..., ::-1].astype(np.float32) / 255.0)
        out.append(to_raw(mosaic(rgb[: rgb.shape[0] // 2 * 2, : rgb.shape[1] // 2 * 2]), rng))
    return np.stack(out), np.round(np.arange(len(out)) * 1e6 / fps).astype(np.int64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=float, default=0.5, help="scene duration (s)")
    ap.add_argument("--video", help="optional mp4 -> data/video_raw.bin")
    ap.add_argument("--bench-frames", type=int, default=210)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if a.video:
        fr, t = video_to_raw(a.video)
        write_frames(f"{DATA}/video_raw.bin", fr, t)
        print("video", fr.shape)
        sys.exit()
    for s in SCENES:
        for fps, noise, tag in [(1000, False, "1000"), (30, False, "30"), (30, True, "30n")]:
            fr, t = render(s, fps, a.dur, noise)
            write_frames(f"{DATA}/{s}_{tag}.bin", fr, t)
            print(f"{s}_{tag}: {fr.shape} raw range [{fr.min()},{fr.max()}]")
    for bw, bh in [(346, 260), (1280, 720)]:  # latency benchmark inputs (texture scene, 30 fps)
        fr, t = render("texture", 30, (a.bench_frames - 1) / 30, False, bw, bh)
        write_frames(f"{DATA}/bench_{bw}x{bh}.bin", fr, t)
        print(f"bench_{bw}x{bh}: {fr.shape}")
