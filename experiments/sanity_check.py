"""Experiment 1 (M1): sanity check of the float reference on synthetic scenes.
Prints event counts, checks polarity on the bars scene, writes results/m1_event_grid.png.
Usage: python -m experiments.sanity_check
"""
import os
for k, v in {"XDG_CACHE_HOME": "./.cache", "MPLCONFIGDIR": "./.cache/mpl",
             "HF_HOME": "./.cache/huggingface", "TORCH_HOME": "./.cache/torch"}.items():
    os.environ.setdefault(k, v)
import numpy as np
from raw2event.scenes import SCENES, BLACK
from raw2event.events import events_for, raw_first_L, bin2x2, accum


def polarity_agreement():
    """Bars move right, so ON events must sit where dL/dx < 0: fraction of GT events with sign(p) == -sign(dL/dx)."""
    ev, raw, t = events_for("bars", "1000", "raw")
    gx = np.gradient(raw_first_L(raw), axis=2)
    g = gx[np.clip(np.searchsorted(t, ev["t"]), 0, len(t) - 1), ev["y"], ev["x"]]
    m = np.abs(g) > 1e-3
    return float(np.mean(np.sign(ev["p"][m]) == -np.sign(g[m]))), int(m.sum())


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("results", exist_ok=True)
    win = (100_000, 200_000)  # us window for the accumulated frames
    fig, ax = plt.subplots(len(SCENES), 4, figsize=(14, 3.2 * len(SCENES)))
    for i, s in enumerate(SCENES):
        cols = [("raw-first 30fps", "30", "raw"), ("ISP-first 30fps", "30", "isp"), ("GT raw-first 1000fps", "1000", "raw")]
        for j, (name, tag, p) in enumerate(cols):
            ev, raw, t = events_for(s, tag, p)
            h, w = raw.shape[1] // 2, raw.shape[2] // 2
            on, off = (ev["p"] > 0).sum(), (ev["p"] < 0).sum()
            print(f"M1 {s:8s} {name:22s} events={ev.size:8d} ON={on:8d} OFF={off:8d}")
            if j == 0:
                ax[i, 0].imshow(bin2x2(raw[len(raw) // 4].astype(float) - BLACK) ** (1 / 2.2), cmap="gray")
                ax[i, 0].set_title(f"{s}: binned raw (display gamma)")
            a = accum(ev, h, w, *win)
            m = max(np.abs(a).max(), 1)
            ax[i, j + 1].imshow(a, cmap="bwr", vmin=-m, vmax=m)
            ax[i, j + 1].set_title(f"{name} (red=ON blue=OFF)\n{win[0]//1000}-{win[1]//1000} ms, n={ev.size}")
    for a_ in ax.ravel():
        a_.axis("off")
    plt.tight_layout()
    plt.savefig("results/m1_event_grid.png", dpi=90)
    print("saved results/m1_event_grid.png")

    agree, n = polarity_agreement()
    print(f"M1 bars polarity agrees with -dL/dx: {agree:.4f} (n={n})")
    assert agree > 0.95, "polarity check failed"


if __name__ == "__main__":
    main()
