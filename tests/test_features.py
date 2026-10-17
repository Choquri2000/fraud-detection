import numpy as np
import pandas as pd

from src.fraud_detection.features import build_features, history_features, make_key

HIST = ["card_txn_count", "card_vel_1h", "card_vel_24h", "card_secs_since_prev", "card_amt_mean", "card_amt_std",
        "card_amt_z", "dev_txn_count", "dev_vel_1h", "dev_vel_24h"]
TOL = dict(check_exact=False, rtol=1e-9, atol=1e-9)


def _sorted(df):
    return df.sort_values(["TransactionDT", "TransactionID"])


def test_hand_computed_example():
    # card a: ids 1 (t=0), 2 (t=1800), 3 (t=10000), 5 (t=90000); card b: id 4
    df = pd.DataFrame({"TransactionID": [3, 1, 2, 4, 5], "k": ["a", "a", "a", "b", "a"],
                       "TransactionDT": [10000, 0, 1800, 50, 90000], "TransactionAmt": [60.0, 10.0, 30.0, 5.0, 20.0]})
    f = history_features(df, "k", "TransactionDT", "TransactionAmt", "TransactionID", "c").set_index(df["TransactionID"])
    assert list(f.loc[[1, 2, 3, 5], "c_txn_count"]) == [0, 1, 2, 3]
    assert f.loc[4, "c_txn_count"] == 0
    assert list(f.loc[[2, 3, 5], "c_vel_1h"]) == [1, 0, 0]       # id 3 at t=10000: nothing within the last 3600 s
    assert list(f.loc[[2, 3, 5], "c_vel_24h"]) == [1, 2, 1]      # id 5 at t=90000: only id 3 (10000) is within 86400 s
    assert np.isnan(f.loc[1, "c_secs_since_prev"]) and f.loc[3, "c_secs_since_prev"] == 8200
    assert np.isnan(f.loc[1, "c_amt_mean"]) and f.loc[3, "c_amt_mean"] == 20.0
    assert np.isclose(f.loc[3, "c_amt_std"], np.std([10, 30], ddof=1))
    assert np.isclose(f.loc[3, "c_amt_z"], (60 - 20) / np.std([10, 30], ddof=1))
    assert f.loc[2, "c_amt_z"] == 0.0                            # one prior transaction: z undefined -> 0


def test_features_unchanged_when_later_rows_removed(raw, cfg):
    full = build_features(raw, cfg)
    s = _sorted(raw)
    cut = s["TransactionDT"].iloc[int(len(s) * 0.6)]
    kept_ids = s.loc[s["TransactionDT"] <= cut, "TransactionID"]
    part = build_features(raw[raw["TransactionID"].isin(kept_ids)], cfg)
    a = full.set_index("TransactionID").loc[part["TransactionID"], HIST]
    b = part.set_index("TransactionID")[HIST]
    pd.testing.assert_frame_equal(a, b, **TOL)


def test_row_features_ignore_all_later_rows_one_by_one(raw, cfg):
    small = raw.sample(600, random_state=0)
    full = build_features(small, cfg).set_index("TransactionID")
    s = _sorted(small)
    for pos in [50, 200, 450]:
        sub = s.iloc[: pos + 1]
        f = build_features(sub, cfg).set_index("TransactionID")
        rid = sub["TransactionID"].iloc[-1]
        pd.testing.assert_series_equal(full.loc[rid, HIST].astype(float), f.loc[rid, HIST].astype(float), **TOL)


def test_changing_later_rows_does_not_change_earlier_features(raw, cfg):
    s = _sorted(raw)
    base = build_features(s, cfg).set_index("TransactionID")
    mod = s.copy()
    last = mod.index[-500:]
    mod.loc[last, "TransactionAmt"] *= 50
    mod.loc[last, "isFraud"] = 1 - mod.loc[last, "isFraud"]
    new = build_features(mod, cfg).set_index("TransactionID")
    early = s["TransactionID"].iloc[:-500]
    pd.testing.assert_frame_equal(base.loc[early, HIST], new.loc[early, HIST], **TOL)


def test_labels_are_not_used(raw, cfg):
    flipped = raw.assign(isFraud=1 - raw["isFraud"])
    pd.testing.assert_frame_equal(build_features(raw, cfg)[HIST], build_features(flipped, cfg)[HIST])


def test_row_order_does_not_matter(raw, cfg):
    a = build_features(raw, cfg)
    b = build_features(raw.sample(frac=1.0, random_state=9), cfg).set_index("TransactionID").loc[raw["TransactionID"]]
    np.testing.assert_allclose(a[HIST].to_numpy(float), b[HIST].to_numpy(float), rtol=1e-9, equal_nan=True)


def test_make_key_handles_missing():
    k = make_key(pd.DataFrame({"a": [1, None], "b": ["x", None]}), ["a", "b"])
    assert k.iloc[0] == "1.0|x" and pd.isna(k.iloc[1])
