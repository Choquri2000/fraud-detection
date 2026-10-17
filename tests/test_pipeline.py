import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.fraud_detection.features import build_features, feature_columns
from src.fraud_detection.models import model_zoo, rule_score
from src.fraud_detection.split import time_split


def _setup(raw, cfg):
    f = build_features(raw, cfg)
    num, cat = feature_columns(f, cfg)
    tr, te = time_split(f, "TransactionDT", 0.7, 7)
    return f, num, cat, tr, te


def test_models_fit_and_beat_prevalence_on_later_period(raw, cfg):
    f, num, cat, tr, te = _setup(raw, cfg)
    y_tr, y_te = f["isFraud"].iloc[tr].to_numpy(), f["isFraud"].iloc[te].to_numpy()
    zoo = model_zoo(cfg, num, cat, y_tr)
    assert set(zoo) == {"logistic", "logistic_balanced", "lightgbm", "lightgbm_spw"}
    means = {}
    for name, pipe in zoo.items():
        pipe.fit(f.iloc[tr][num + cat], y_tr)
        p = pipe.predict_proba(f.iloc[te][num + cat])[:, 1]
        means[name] = p.mean()
        assert average_precision_score(y_te, p) > 3 * y_te.mean(), name
    # class weighting shifts probabilities upwards: the reason calibration is checked
    assert means["logistic_balanced"] > 2 * means["logistic"]


def test_unseen_categories_at_test_time_do_not_crash(raw, cfg):
    f, num, cat, tr, te = _setup(raw, cfg)
    pipe = model_zoo(cfg, num, cat, f["isFraud"].iloc[tr].to_numpy())["lightgbm"]
    pipe.fit(f.iloc[tr][num + cat], f["isFraud"].iloc[tr])
    X = f.iloc[te][num + cat].copy()
    X["P_emaildomain"] = "never-seen.example"
    assert np.isfinite(pipe.predict_proba(X)[:, 1]).all()


def test_rule_score_beats_prevalence(raw, cfg):
    f = build_features(raw, cfg)
    s = rule_score(f, cfg["rule_baseline"])
    assert average_precision_score(f["isFraud"], s) > f["isFraud"].mean()


def test_end_to_end_run_writes_reports(tmp_path, monkeypatch):
    import run_experiment

    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--synthetic", "--rows", "6000", "--out", str(tmp_path)])
    run_experiment.main()
    for name in ["metrics.csv", "budget_table.csv", "drift.csv", "cost_curve.csv", "bucket_performance.csv",
                 "shap_importance.csv", "shap_summary.png", "pr_curves.png", "calibration.png", "cost_curve.png"]:
        assert (tmp_path / name).exists(), name
    m = pd.read_csv(tmp_path / "metrics.csv")
    assert {"rule", "logistic", "lightgbm", "lightgbm_spw"} <= set(m["model"])
