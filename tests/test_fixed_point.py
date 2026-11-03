"""Fixed-point C++ generator vs float reference on a short synthetic clip (skipped if the binary is not built)."""
import os
import numpy as np
import pytest
from raw2event.scenes import render, write_frames
from raw2event.events import raw_first_L, gen_events
from raw2event.native import EXE, run_cpp
from raw2event import metrics as M

pytestmark = pytest.mark.skipif(not os.path.exists(EXE), reason="build/event_fx not built")


@pytest.mark.parametrize("fl,lb,interp,min_f1", [(16, 12, 1, 0.999), (10, 8, 1, 0.99)])
def test_fixed_matches_float(tmp_path, fl, lb, interp, min_f1):
    raw, t = render("bars", 30, 0.2)
    write_frames(tmp_path / "f.bin", raw, t)
    ev, stats = run_cpp(str(tmp_path / "f.bin"), fl, lb, interp, str(tmp_path / "e.bin"))
    ref = gen_events(raw_first_L(raw), t)
    assert stats["events"] == len(ev) > 0
    assert M.prf(ev, ref, raw.shape[2] // 2, 0.5 * 1e6 / 30)[2] >= min_f1
