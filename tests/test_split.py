import numpy as np
import pytest

from src.fraud_detection.split import assert_time_ordered, time_split


def test_no_test_row_precedes_any_train_row(raw):
    tr, te = time_split(raw, "TransactionDT", 0.7, 7)
    t = raw["TransactionDT"].to_numpy()
    assert t[te].min() > t[tr].max()
    assert len(set(tr) & set(te)) == 0


def test_gap_is_respected_and_rows_dropped(raw):
    tr, te = time_split(raw, "TransactionDT", 0.7, 7)
    t = raw["TransactionDT"].to_numpy()
    assert t[te].min() - t[tr].max() >= 7 * 86400 - 1
    assert len(tr) + len(te) < len(raw)


def test_split_is_independent_of_row_order(raw):
    shuf = raw.sample(frac=1.0, random_state=5).reset_index(drop=True)
    tr, te = time_split(shuf, "TransactionDT", 0.6, 3)
    t = shuf["TransactionDT"].to_numpy()
    assert t[te].min() > t[tr].max()
    assert t[tr].max() <= np.quantile(raw["TransactionDT"], 0.6)


def test_guard_raises_on_overlap_and_bad_args(raw):
    with pytest.raises(AssertionError):
        assert_time_ordered(np.array([1, 5, 9]), np.array([8, 12]))
    with pytest.raises(ValueError):
        time_split(raw, "TransactionDT", 1.2, 1)
    with pytest.raises(ValueError):
        time_split(raw, "TransactionDT", 0.9, 400)
