"""SHAP for the LightGBM pipeline."""
import numpy as np
import pandas as pd


def tree_feature_names(pipe):
    return list(pipe.named_steps["prep"].get_feature_names_out())


def shap_summary(pipe, X, sample=2000, seed=0):
    """Return (shap_values, transformed_frame, importance_df) for a fitted LightGBM pipeline."""
    import shap

    Xs = X.sample(min(sample, len(X)), random_state=seed)
    Z = pipe.named_steps["prep"].transform(Xs)
    names = [n.split("__", 1)[-1] for n in tree_feature_names(pipe)]
    Zdf = pd.DataFrame(np.asarray(Z, dtype=float), columns=names)
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Zdf)
    if isinstance(sv, list):   # older shap versions return one array per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False)
    return sv, Zdf, imp
