"""Metrics for a capacity-constrained fraud team: alert budgets, expected cost, drift, decay."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ranking(score, seed=0):
    """Indices from highest to lowest score. Ties are broken randomly (not by row order), which matters
    for the rule baseline where many rows share a score."""
    rng = np.random.default_rng(seed)
    return np.lexsort((rng.random(len(score)), -np.asarray(score, dtype=float)))


def n_alerts(n, budget):
    return max(int(round(n * budget)), 1)


def budget_metrics(y, score, budget, seed=0):
    """Review the top `budget` share of transactions."""
    y = np.asarray(y)
    k = n_alerts(len(y), budget)
    top = ranking(score, seed)[:k]
    caught = int(y[top].sum())
    total = int(y.sum())
    return {"budget": budget, "alerts": k, "caught": caught,
            "recall": caught / total if total else np.nan, "precision": caught / k,
            "lift": (caught / k) / y.mean() if y.mean() > 0 else np.nan}


def missed_fraud_loss(amount, costs):
    if costs["missed_fraud_mode"] == "amount":
        return costs["missed_fraud_multiplier"] * np.asarray(amount, dtype=float)
    return np.full(len(amount), float(costs["missed_fraud_fixed"]))


def cost_curve(y, score, amount, costs, max_budget=0.10, points=100, seed=0):
    """Expected cost of reviewing the top k transactions: review_cost * k + loss on fraud not reviewed."""
    y = np.asarray(y)
    order = ranking(score, seed)
    loss = missed_fraud_loss(amount, costs) * y
    caught = np.cumsum(loss[order])
    total_loss = loss.sum()
    n = len(y)
    ks = np.unique(np.linspace(0, n_alerts(n, max_budget), points).astype(int))
    cum = np.r_[0.0, caught][ks]
    cost = costs["review_cost"] * ks + (total_loss - cum)
    return pd.DataFrame({"budget": ks / n, "alerts": ks, "cost": cost, "cost_no_review": total_loss,
                         "saving_vs_no_review": total_loss - cost})


def calibration_table(y, p, bins=10):
    y = np.asarray(y)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-12
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    t = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"),
                                                                      n=("y", "size")).reset_index()
    return t


def expected_calibration_error(y, p, bins=10):
    t = calibration_table(y, p, bins)
    return float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["n"].sum())


def psi(reference, current, bins=10, eps=1e-4):
    """Population stability index with quantile bins fixed on the reference sample."""
    ref = pd.Series(reference).dropna().to_numpy()
    cur = pd.Series(current).dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def overall_metrics(y, score, amount, cfg_eval, seed=0, probabilistic=True):
    y = np.asarray(y)
    row = {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
           "prevalence": y.mean()}
    if probabilistic:
        row["brier"] = brier_score_loss(y, score)
        row["ece"] = expected_calibration_error(y, score)
    for b in cfg_eval["alert_budgets"]:
        bm = budget_metrics(y, score, b, seed)
        row[f"recall@{b:.1%}"] = bm["recall"]
        row[f"precision@{b:.1%}"] = bm["precision"]
    cc = cost_curve(y, score, amount, cfg_eval["costs"], cfg_eval["cost_curve_max_budget"], seed=seed)
    best = cc.loc[cc["cost"].idxmin()]
    row["min_cost"] = best["cost"]
    row["min_cost_budget"] = best["budget"]
    row["cost_no_review"] = best["cost_no_review"]
    return row


def time_buckets(t, n_buckets):
    """Equal-width buckets over the time range of `t`; returns integer bucket id per row."""
    edges = np.linspace(np.min(t), np.max(t), n_buckets + 1)
    return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_buckets - 1)


def bucket_performance(y, scores, t, n_buckets, budget, seed=0):
    """Per-bucket PR-AUC and recall at the alert budget, applied WITHIN each bucket."""
    y = np.asarray(y)
    b = time_buckets(t, n_buckets)
    rows = []
    for name, s in scores.items():
        s = np.asarray(s)
        for i in range(n_buckets):
            m = b == i
            if m.sum() == 0 or y[m].sum() == 0:
                continue
            rows.append({"model": name, "bucket": i, "n": int(m.sum()), "fraud_rate": y[m].mean(),
                         "pr_auc": average_precision_score(y[m], s[m]),
                         f"recall@{budget:.1%}": budget_metrics(y[m], s[m], budget, seed)["recall"]})
    return pd.DataFrame(rows)


def drift_table(train_scores, test_scores, t_test, n_buckets, train_feats=None, test_feats=None):
    """PSI of model scores and key features: train vs each test bucket, and first test bucket vs later ones.
    Train scores are in-sample (optimistic), hence the second reference."""
    b = time_buckets(t_test, n_buckets)
    rows = []

    def add(kind, name, ref_name, ref, cur_all):
        for i in range(n_buckets):
            rows.append({"kind": kind, "name": name, "reference": ref_name, "bucket": i,
                         "psi": psi(ref, np.asarray(cur_all)[b == i])})

    for name, s in test_scores.items():
        add("score", name, "train", train_scores[name], s)
        add("score", name, "first_test_bucket", np.asarray(s)[b == 0], s)
    if train_feats is not None:
        for c in train_feats.columns:
            add("feature", c, "train", train_feats[c].to_numpy(), test_feats[c].to_numpy())
    return pd.DataFrame(rows)
