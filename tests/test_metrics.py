import numpy as np
from raw2event import metrics as M

E = np.dtype([("x", "<u2"), ("y", "<u2"), ("t", "<f8"), ("p", "i1")])


def test_match_rules():
    ref = np.array([(1, 1, 100, 1), (1, 1, 200, 1), (2, 1, 100, -1)], E)
    pred = np.array([(1, 1, 150, 1), (1, 1, 900, 1), (2, 1, 100, 1)], E)  # 1 match; 1 too late; 1 wrong polarity
    assert M.match_count(pred, ref, 10, 60) == 1


def test_identical_streams_are_perfect():
    ref = np.array([(1, 1, 100, 1), (3, 2, 50, -1)], E)
    assert M.prf(ref, ref, 10, 0)[2] == 1.0
    assert M.count_rmse(ref, ref, 5, 10) == 0.0
