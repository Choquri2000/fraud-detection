"""Rule baseline, logistic regression and LightGBM, with two ways of handling imbalance (no SMOTE)."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import make_preprocessor


def rule_score(feats, rule_cfg):
    """Hand-set rule, not fitted on labels: unusual amount for this card + burst + unseen card."""
    z = np.clip(feats["card_amt_z"].fillna(0).to_numpy(), 0, 5)
    vel = np.clip(feats["card_vel_1h"].fillna(0).to_numpy(), 0, 5)
    new = (feats["card_txn_count"].fillna(0).to_numpy() == 0).astype(float)
    return rule_cfg["w_amount_z"] * z + rule_cfg["w_velocity_1h"] * vel + rule_cfg["w_new_card"] * new


def model_zoo(cfg, num_cols, cat_cols, y_train):
    """Return {name: unfitted sklearn Pipeline}. Imbalance: none vs class weight / scale_pos_weight."""
    m = cfg["modeling"]
    k = cfg["features"]["top_k_categories"]
    seed = m["seed"]
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    lgb = dict(m["lightgbm"], random_state=seed, verbose=-1)

    def lr(weight):
        return Pipeline([("prep", make_preprocessor(num_cols, cat_cols, "linear", k)),
                         ("clf", LogisticRegression(C=m["logistic"]["C"], max_iter=m["logistic"]["max_iter"],
                                                    class_weight=weight))])

    def gbm(**kw):
        return Pipeline([("prep", make_preprocessor(num_cols, cat_cols, "tree", k)),
                         ("clf", LGBMClassifier(**lgb, **kw))])

    return {
        "logistic": lr(None),
        "logistic_balanced": lr("balanced"),
        "lightgbm": gbm(),
        "lightgbm_spw": gbm(scale_pos_weight=spw),
    }
