"""Loading and joining the two IEEE-CIS tables."""
import pandas as pd


def load_raw(cfg):
    d = cfg["data"]
    tx = pd.read_csv(d["transaction_path"])
    ident = None
    if d.get("identity_path"):
        try:
            ident = pd.read_csv(d["identity_path"])
        except FileNotFoundError:
            ident = None
    return tx, ident


def join_tables(tx, ident, id_col="TransactionID"):
    """Left join: most transactions have no identity row (that absence is itself informative)."""
    if ident is None:
        out = tx.copy()
        out["has_identity"] = 0
        return out
    if ident[id_col].duplicated().any():
        raise ValueError("identity table has duplicate TransactionID")
    out = tx.merge(ident, on=id_col, how="left")
    out["has_identity"] = out[id_col].isin(ident[id_col]).astype(int)
    return out


def fraud_summary(df, target, time_col):
    return {
        "rows": int(len(df)),
        "fraud_rate": float(df[target].mean()),
        "days": float((df[time_col].max() - df[time_col].min()) / 86400),
    }
