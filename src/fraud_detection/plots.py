"""Matplotlib figures written to reports/ (Agg backend, no display needed)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from .evaluate import calibration_table, ranking


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pr_curves(y, scores, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=name)
    ax.axhline(np.mean(y), color="grey", ls=":", label="prevalence")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall (time-ordered test)")
    ax.legend(fontsize=8)
    _save(fig, path)


def budget_recall(y, scores, budgets_max, path, seed=0):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.linspace(0.001, budgets_max, 60)
    total = np.sum(y)
    for name, s in scores.items():
        order = ranking(s, seed)
        cum = np.cumsum(np.asarray(y)[order])
        ax.plot(xs * 100, [cum[max(int(round(len(y) * x)), 1) - 1] / total for x in xs], label=name)
    ax.set(xlabel="alert budget (% of transactions reviewed)", ylabel="fraud recall",
           title="Recall vs alert budget")
    ax.legend(fontsize=8)
    _save(fig, path)


def cost_curves(curves, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, c in curves.items():
        ax.plot(c["budget"] * 100, c["cost"], label=name)
    first = next(iter(curves.values()))
    ax.axhline(first["cost_no_review"].iloc[0], color="grey", ls=":", label="no review")
    ax.set(xlabel="alert budget (%)", ylabel="expected cost (assumed units)", title="Expected cost vs budget")
    ax.legend(fontsize=8)
    _save(fig, path)


def calibration(y, scores, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    hi = 0.0
    for name, p in scores.items():
        t = calibration_table(y, p)
        ax.plot(t["mean_pred"], t["frac_pos"], marker="o", ms=3, label=name)
        hi = max(hi, t["mean_pred"].max(), t["frac_pos"].max())
    ax.plot([0, hi], [0, hi], "k:", label="perfect")
    ax.set(xlabel="mean predicted probability", ylabel="observed fraud rate", title="Calibration (decile bins)")
    ax.legend(fontsize=8)
    _save(fig, path)


def decay(perf, metric, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, g in perf.groupby("model"):
        ax.plot(g["bucket"], g[metric], marker="o", label=name)
    ax.set(xlabel="test time bucket (0 = right after the gap)", ylabel=metric, title="Performance decay over time")
    ax.legend(fontsize=8)
    _save(fig, path)


def psi_over_time(drift, path):
    d = drift[(drift["kind"] == "score") & (drift["reference"] == "train")]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, g in d.groupby("name"):
        ax.plot(g["bucket"], g["psi"], marker="o", label=name)
    ax.axhline(0.25, color="grey", ls=":")
    ax.set(xlabel="test time bucket", ylabel="PSI vs train scores", title="Score drift (0.25 = common alarm level)")
    ax.legend(fontsize=8)
    _save(fig, path)


def shap_summary_plot(sv, Zdf, path):
    import shap

    shap.summary_plot(sv, Zdf, show=False, max_display=20)
    fig = plt.gcf()
    _save(fig, path)
