"""Time-split fraud experiment: features, models, alert-budget evaluation, drift, SHAP -> reports/.

    python run_experiment.py                  # real IEEE-CIS files from configs/config.yaml
    python run_experiment.py --synthetic      # synthetic data in the same layout (smoke test)
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.fraud_detection import evaluate as ev
from src.fraud_detection import plots
from src.fraud_detection.data import fraud_summary, join_tables, load_raw
from src.fraud_detection.explain import shap_summary
from src.fraud_detection.features import build_features, feature_columns
from src.fraud_detection.models import model_zoo, rule_score
from src.fraud_detection.split import time_split
from src.fraud_detection.synthetic import make_transactions

log = logging.getLogger("experiment")
DRIFT_FEATURES = ["TransactionAmt", "card_amt_z", "card_vel_24h", "card_secs_since_prev", "C1", "D1"]


def load(cfg, synthetic, rows, seed):
    if synthetic:
        tx, ident = make_transactions(rows or 40000, seed=seed)
    else:
        tx, ident = load_raw(cfg)
        if rows:
            tx = tx.sort_values(cfg["data"]["time_col"]).head(rows)   # earliest rows only, never a random sample
    return join_tables(tx, ident, cfg["data"]["id_col"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = yaml.safe_load(open(args.config, encoding="utf8"))
    seed = cfg["modeling"]["seed"]
    d, ecfg = cfg["data"], cfg["evaluation"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load(cfg, args.synthetic, args.rows, seed)
    log.info("data: %s", fraud_summary(raw, d["target"], d["time_col"]))
    feats = build_features(raw, cfg)
    num_cols, cat_cols = feature_columns(feats, cfg)

    tr, te = time_split(feats, d["time_col"], cfg["split"]["train_frac"], cfg["split"]["gap_days"])
    train, test = feats.iloc[tr], feats.iloc[te]
    y_tr, y_te = train[d["target"]].to_numpy(), test[d["target"]].to_numpy()
    amt_te, t_te = test[d["amount_col"]].to_numpy(), test[d["time_col"]].to_numpy()
    log.info("train %d rows (fraud %.2f%%), test %d rows (fraud %.2f%%)", len(train), 100 * y_tr.mean(),
             len(test), 100 * y_te.mean())

    # ---- fit models; scores are only ever evaluated on the later test period ----
    test_scores, train_scores = {"rule": rule_score(test, cfg["rule_baseline"])}, {"rule": rule_score(train, cfg["rule_baseline"])}
    fitted = {}
    for name, pipe in model_zoo(cfg, num_cols, cat_cols, y_tr).items():
        log.info("fitting %s", name)
        pipe.fit(train[num_cols + cat_cols], y_tr)
        fitted[name] = pipe
        test_scores[name] = pipe.predict_proba(test[num_cols + cat_cols])[:, 1]
        train_scores[name] = pipe.predict_proba(train[num_cols + cat_cols])[:, 1]

    # ---- metrics ----
    rows, budget_rows, curves = [], [], {}
    for name, s in test_scores.items():
        row = {"model": name, **ev.overall_metrics(y_te, s, amt_te, ecfg, seed, probabilistic=name != "rule")}
        rows.append(row)
        for b in ecfg["alert_budgets"]:
            budget_rows.append({"model": name, **ev.budget_metrics(y_te, s, b, seed)})
        curves[name] = ev.cost_curve(y_te, s, amt_te, ecfg["costs"], ecfg["cost_curve_max_budget"], seed=seed)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(out / "budget_table.csv", index=False)
    pd.concat([c.assign(model=n) for n, c in curves.items()]).to_csv(out / "cost_curve.csv", index=False)

    # ---- drift and decay ----
    nb = cfg["split"]["n_buckets"]
    drift = ev.drift_table(train_scores, test_scores, t_te, nb, train[DRIFT_FEATURES], test[DRIFT_FEATURES])
    drift.to_csv(out / "drift.csv", index=False)
    perf = ev.bucket_performance(y_te, test_scores, t_te, nb, ecfg["alert_budgets"][1], seed)
    perf.to_csv(out / "bucket_performance.csv", index=False)

    # ---- plots ----
    plots.pr_curves(y_te, test_scores, out / "pr_curves.png")
    plots.budget_recall(y_te, test_scores, ecfg["cost_curve_max_budget"], out / "budget_recall.png", seed)
    plots.cost_curves(curves, out / "cost_curve.png")
    plots.calibration(y_te, {k: v for k, v in test_scores.items() if k != "rule"}, out / "calibration.png")
    if len(perf):
        plots.decay(perf, f"recall@{ecfg['alert_budgets'][1]:.1%}", out / "decay_recall.png")
    plots.psi_over_time(drift, out / "score_psi.png")

    # ---- SHAP ----
    sname = cfg["modeling"]["shap_model"]
    sv, Zdf, imp = shap_summary(fitted[sname], test[num_cols + cat_cols], cfg["modeling"]["shap_sample"], seed)
    imp.to_csv(out / "shap_importance.csv", index=False)
    plots.shap_summary_plot(sv, Zdf, out / "shap_summary.png")

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(rows).set_index("model"))
    log.info("wrote %s", out.resolve())


if __name__ == "__main__":
    main()
