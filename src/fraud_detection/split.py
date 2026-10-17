"""Strict time-ordered split. There is deliberately no random split in this project."""
import numpy as np


def time_split(df, time_col, train_frac=0.7, gap_days=7.0):
    """Return (train_idx, test_idx) as positional index arrays.

    train: time <= cutoff, where cutoff is the `train_frac` quantile of the time column
    gap:   (cutoff, cutoff + gap_days] is dropped; it stands in for chargeback/label delay
    test:  time > cutoff + gap_days
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    t = df[time_col].to_numpy()
    cutoff = np.quantile(t, train_frac)
    test_start = cutoff + gap_days * 86400
    train_idx = np.flatnonzero(t <= cutoff)
    test_idx = np.flatnonzero(t > test_start)
    assert_time_ordered(t[train_idx], t[test_idx])
    return train_idx, test_idx


def assert_time_ordered(train_t, test_t):
    if len(train_t) == 0 or len(test_t) == 0:
        raise ValueError("empty train or test period; check train_frac / gap_days")
    if test_t.min() <= train_t.max():
        raise AssertionError("time leakage: a test transaction is not later than every train transaction")
