"""Loading and joining the two IEEE-CIS tables."""
import pandas as pd

KEEP_FLOAT64 = {"TransactionDT", "TransactionAmt"}  # keep exact values for time ordering and amounts


def read_csv_compact(path, chunksize=100_000, usecols=None):
    """Read a wide CSV in chunks, storing float columns as float32 (about half the memory).

    The transaction file has ~394 columns and 590k rows; as float64 it needs ~2 GB, which can
    exhaust a laptop once joins and feature copies are added.
    """
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        f64 = [c for c in chunk.select_dtypes("float64").columns if c not in KEEP_FLOAT64]
        chunk[f64] = chunk[f64].astype("float32")
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def load_raw(cfg):
    """Read only the columns the pipeline uses; the id column is always kept for the join."""
    from .features import needed_columns

    d = cfg["data"]
    wanted = needed_columns(cfg)
    keep = lambda c: c in wanted  # noqa: E731
    tx = read_csv_compact(d["transaction_path"], usecols=keep)
    ident = None
    if d.get("identity_path"):
        try:
            ident = read_csv_compact(d["identity_path"], usecols=keep)
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
