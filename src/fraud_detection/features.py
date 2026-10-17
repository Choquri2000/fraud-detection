"""Point-in-time behavioural features and model preprocessors.

Every history feature for a row uses only rows that come strictly EARLIER in the order
(time, TransactionID). Labels are never used. Rows are never dropped; later rows cannot
change an earlier row's features (see tests/test_features.py).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

NUM_BASE = (["TransactionAmt", "dist1", "addr1", "addr2", "has_identity", "hour"]
            + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]
            + ["id_01", "id_02"])
HISTORY_PREFIXES = ("card_txn", "card_vel", "card_secs", "card_amt", "dev_txn", "dev_vel")
CAT_BASE = (["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
             "id_30", "id_31", "id_33"] + [f"M{i}" for i in range(1, 10)])


def needed_columns(cfg):
    """Raw columns the pipeline actually uses (so the loader can skip the other ~350 V/id columns)."""
    d, f = cfg["data"], cfg["features"]
    cols = {d["id_col"], d["time_col"], d["target"], d["amount_col"]}
    cols |= set(NUM_BASE) | set(CAT_BASE) | set(f["v_columns"]) | set(f["card_key"])
    if f.get("device_key"):
        cols.add(f["device_key"])
    return cols


def make_key(df, cols):
    """Join the key columns into one string; rows where ALL parts are missing get NaN."""
    cols = [cols] if isinstance(cols, str) else list(cols)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=object)
    parts = df[cols].astype(str).where(df[cols].notna(), "NA")
    key = parts.agg("|".join, axis=1) if len(cols) > 1 else parts.iloc[:, 0]
    return key.where(df[cols].notna().any(axis=1))


def history_features(df, key, time_col, amount_col, id_col, prefix, windows=(3600, 86400), with_amount=True):
    """Features of each row from earlier rows of the same key. Returns a frame aligned to df.index."""
    order = np.lexsort((df[id_col].to_numpy(), df[time_col].to_numpy()))
    d = df.iloc[order]
    valid = d[key].notna().to_numpy()
    d = d[valid]
    gcode = pd.factorize(d[key])[0]
    o2 = np.argsort(gcode, kind="stable")        # stable: keeps time order inside each group
    d = d.iloc[o2]
    g = gcode[o2]
    t = d[time_col].to_numpy().astype(np.int64)
    a = d[amount_col].to_numpy(dtype=float)
    n = len(d)
    pos = np.arange(n)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]) if n else np.array([], dtype=int)
    first = np.repeat(starts, np.diff(np.r_[starts, n])) if n else np.array([], dtype=int)
    n_prior = pos - first

    out = {f"{prefix}_txn_count": n_prior.astype(float)}
    comp = g.astype(np.int64) * (2 ** 32) + (t - t.min() if n else t)   # monotone (group, time) key
    for w in windows:
        out[f"{prefix}_vel_{w // 3600}h"] = (pos - np.searchsorted(comp, comp - w, side="left")).astype(float)
    prev_t = np.r_[0, t[:-1]] if n else t
    out[f"{prefix}_secs_since_prev"] = np.where(n_prior > 0, t - prev_t, np.nan)

    if with_amount:
        ser = pd.Series(a)
        s1 = ser.groupby(g).cumsum().to_numpy() - a
        s2 = (ser ** 2).groupby(g).cumsum().to_numpy() - a ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(n_prior > 0, s1 / n_prior, np.nan)
            var = np.where(n_prior > 1, (s2 - s1 ** 2 / n_prior) / (n_prior - 1), np.nan)
            std = np.sqrt(np.clip(var, 0, None))
            z = np.where((n_prior > 1) & (std > 0), (a - mean) / std, 0.0)
        out[f"{prefix}_amt_mean"] = mean
        out[f"{prefix}_amt_std"] = std
        out[f"{prefix}_amt_z"] = z

    res = pd.DataFrame(out, index=d.index)
    return res.reindex(df.index)


def build_features(df, cfg):
    """Row-level frame = raw columns + hour + point-in-time card/device history features."""
    dcfg, fcfg = cfg["data"], cfg["features"]
    tcol, acol, idc = dcfg["time_col"], dcfg["amount_col"], dcfg["id_col"]
    out = df.copy()
    out["hour"] = (out[tcol] // 3600) % 24
    out["_card"] = make_key(out, fcfg["card_key"])
    wins = tuple(fcfg["velocity_windows_seconds"])
    card = history_features(out, "_card", tcol, acol, idc, "card", wins, with_amount=True)
    out = pd.concat([out, card], axis=1)
    if fcfg.get("device_key") and fcfg["device_key"] in out.columns:
        out["_dev"] = out[fcfg["device_key"]]
        dev = history_features(out, "_dev", tcol, acol, idc, "dev", wins, with_amount=False)
        out = pd.concat([out, dev[[c for c in dev.columns if "secs" not in c]]], axis=1)
    return out.drop(columns=[c for c in ("_card", "_dev") if c in out.columns])


def feature_columns(df, cfg):
    hist = [c for c in df.columns if c.startswith(HISTORY_PREFIXES)]
    num = [c for c in NUM_BASE + list(cfg["features"]["v_columns"]) + hist if c in df.columns]
    cat = [c for c in CAT_BASE if c in df.columns]
    return list(dict.fromkeys(num)), cat


class TopKCategories(BaseEstimator, TransformerMixin):
    """Keep the k most frequent values seen in training; the rest become "__other__", NaN "__missing__"."""

    def __init__(self, k=25):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.keep_ = {c: set(X[c].value_counts().head(self.k).index) for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for c in self.columns_:
            col = X[c]
            out[c] = np.where(col.isna(), "__missing__", np.where(col.isin(self.keep_[c]), col.astype(str), "__other__"))
        return out

    def get_feature_names_out(self, names=None):
        return np.array(self.columns_)


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(num_cols, cat_cols, kind, top_k=25):
    """kind='linear': impute+signed-log+scale numerics, one-hot categoricals.
    kind='tree': numerics as is (NaN kept), ordinal-coded categoricals."""
    if kind == "linear":
        num = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("log", FunctionTransformer(_signed_log1p)), ("sc", StandardScaler())])
        cat = Pipeline([("top", TopKCategories(top_k)),
                        ("oh", OneHotEncoder(handle_unknown="ignore", dtype=np.float32))])
    else:
        num = "passthrough"
        cat = Pipeline([("top", TopKCategories(top_k)),
                        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])
