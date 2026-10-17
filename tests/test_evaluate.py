import numpy as np

from src.fraud_detection import evaluate as ev

COSTS = {"review_cost": 5.0, "missed_fraud_mode": "amount", "missed_fraud_multiplier": 1.0, "missed_fraud_fixed": 100.0}


def test_budget_metrics_perfect():
    y = np.zeros(1000, int)
    y[:20] = 1
    m = ev.budget_metrics(y, y.astype(float), 0.02)
    assert m["alerts"] == 20 and m["recall"] == 1.0 and m["precision"] == 1.0


def test_ties_are_not_broken_by_row_order():
    y = np.r_[np.ones(50), np.zeros(950)].astype(int)       # frauds first in row order
    assert ev.budget_metrics(y, np.zeros(1000), 0.05)["recall"] < 0.5


def test_cost_curve():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.03).astype(int)
    amt = rng.uniform(10, 200, 5000)
    s = y + rng.normal(0, 0.3, 5000)
    c = ev.cost_curve(y, s, amt, COSTS, 0.1)
    assert c["cost"].iloc[0] == c["cost_no_review"].iloc[0]
    assert c["cost"].min() < c["cost_no_review"].iloc[0]
    fixed = ev.cost_curve(y, s, amt, {**COSTS, "missed_fraud_mode": "fixed"}, 0.1)
    assert np.isclose(fixed["cost_no_review"].iloc[0], 100.0 * y.sum())


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    assert ev.psi(a, rng.normal(0, 1, 20000)) < 0.02
    assert ev.psi(a, rng.normal(1.5, 1, 20000)) > 0.25
    assert ev.psi(a, a) < 1e-6


def test_calibration_error():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.2, 100000)
    y = (rng.random(100000) < p).astype(int)
    assert ev.expected_calibration_error(y, p) < 0.01
    assert ev.expected_calibration_error(y, np.clip(p * 3, 0, 1)) > 0.05


def test_buckets_and_decay_table():
    t = np.arange(1000)
    assert list(np.unique(ev.time_buckets(t, 4))) == [0, 1, 2, 3]
    y = np.tile([1, 0, 0, 0], 250)
    perf = ev.bucket_performance(y, {"m": y.astype(float)}, t, 4, 0.1)
    assert len(perf) == 4 and (perf["pr_auc"] == 1.0).all()


def test_drift_table_flags_shifted_bucket():
    rng = np.random.default_rng(0)
    t = np.arange(4000)
    s = rng.normal(0, 1, 4000) + np.where(t > 3000, 2.0, 0.0)
    d = ev.drift_table({"m": rng.normal(0, 1, 4000)}, {"m": s}, t, 4)
    ref = d[d["reference"] == "train"].set_index("bucket")["psi"]
    assert ref[3] > 0.25 > ref[0]
