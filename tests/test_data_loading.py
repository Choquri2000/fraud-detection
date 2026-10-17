import pandas as pd
import yaml
from pathlib import Path

from src.fraud_detection.data import load_raw
from src.fraud_detection.features import needed_columns

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text(encoding="utf8"))


def test_loader_reads_only_needed_columns(tmp_path):
    cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "card1", "V1", "V300", "V339", "C1"]
    tx = pd.DataFrame([[1, 0, 100, 10.5, 111, 0.1, 0.2, 0.3, 1.0],
                       [2, 1, 200, 99.0, 111, 0.4, 0.5, 0.6, 2.0]], columns=cols)
    ident = pd.DataFrame({"TransactionID": [1], "id_01": [-5.0], "id_99": [1.0]})
    tx.to_csv(tmp_path / "t.csv", index=False)
    ident.to_csv(tmp_path / "i.csv", index=False)
    cfg = {**CFG, "data": {**CFG["data"], "transaction_path": str(tmp_path / "t.csv"),
                           "identity_path": str(tmp_path / "i.csv")}}
    t, i = load_raw(cfg)
    assert "V300" not in t.columns and "V339" not in t.columns   # not in config's v_columns
    assert {"V1", "C1", "TransactionID", "isFraud"} <= set(t.columns)
    assert "id_99" not in i.columns and "id_01" in i.columns
    assert t["TransactionAmt"].dtype == "float64"                 # amounts stay exact
    assert "TransactionAmt" in needed_columns(CFG)
