"""Bridge to the fixed-point C++ event generator (build/event_fx)."""
import os
import subprocess
import numpy as np
import pandas as pd

EXE = os.path.join("build", "event_fx.exe" if os.name == "nt" else "event_fx")
EV_I = np.dtype([("x", "<u2"), ("y", "<u2"), ("t", "<i4"), ("p", "i1")])  # packed 9-byte event records
RAWDIR = "results/raw"  # scratch files (git-ignored)


def run_cpp(binfile, fl, lb, interp, out=None, extra=()):
    """Run event_fx on a frames.bin file -> (events or None, stats dict parsed from its key=value stdout line)."""
    cmd = [EXE, "--in", binfile, "--fl", str(fl), "--lut-bits", str(lb), "--interp", str(interp), *map(str, extra)]
    if out:
        cmd += ["--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    stats = dict(kv.split("=") for kv in r.stdout.split())
    ev = np.fromfile(out, EV_I) if out else None
    return ev, {k: float(v) for k, v in stats.items()}


def best_cfg(summary="results/m2_summary.csv"):
    """Smallest-cost (FL, LUT_BITS, interp) whose min-over-scenes F1 >= 0.99 in the bit-width sweep."""
    summ = pd.read_csv(summary)
    ok = summ[summ.f1_min >= 0.99]
    r = ok.iloc[0] if len(ok) else summ.sort_values("f1_mean").iloc[-1]
    return int(r.fl), int(r.lut_bits), int(r.interp)
