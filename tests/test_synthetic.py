from src.fraud_detection.synthetic import make_transactions


def test_layout_and_rate(tables):
    tx, ident = tables
    for c in ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card6", "addr1",
              "addr2", "dist1", "P_emaildomain", "R_emaildomain", "C1", "C14", "D1", "D15", "M1", "M9", "V1", "V12"]:
        assert c in tx.columns
    assert 0.03 < tx["isFraud"].mean() < 0.04
    assert tx["TransactionID"].is_unique and ident["TransactionID"].is_unique
    assert set(ident["TransactionID"]) <= set(tx["TransactionID"])
    assert {"DeviceType", "DeviceInfo", "id_30"} <= set(ident.columns)
    assert 0.1 < len(ident) / len(tx) < 0.5


def test_rows_are_shuffled_not_time_sorted(tables):
    tx, _ = tables
    assert not tx["TransactionDT"].is_monotonic_increasing


def test_reproducible():
    a, _ = make_transactions(n=1000, seed=3)
    b, _ = make_transactions(n=1000, seed=3)
    assert a.equals(b)


def test_temporal_drift_in_risky_domains(tables):
    tx, _ = tables
    f = tx[tx["isFraud"] == 1]
    early = f[f["TransactionDT"] < f["TransactionDT"].quantile(0.5)]["P_emaildomain"]
    late = f[f["TransactionDT"] > f["TransactionDT"].quantile(0.8)]["P_emaildomain"]
    assert (early == "protonmail.com").mean() > 0.05 and (late == "protonmail.com").sum() == 0
    assert (late == "mail.com").mean() > 0.05


def test_fraud_rings_burst_within_two_hours(raw):
    f = raw[raw["isFraud"] == 1]
    ring = f[f["DeviceInfo"].astype(str).str.startswith("RING-")]
    assert len(ring) > 20
    span = ring.groupby("DeviceInfo")["TransactionDT"].agg(lambda s: s.max() - s.min())
    assert span.max() <= 7200
