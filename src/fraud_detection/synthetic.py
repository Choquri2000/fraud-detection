"""Synthetic transactions in the IEEE-CIS column layout (subset), for tests and smoke runs.

Built so the time logic can be tested:
  * fraud is ~`fraud_rate` of rows;
  * fraud rings: bursts on a couple of victim cards sharing a device string and a risky e-mail domain;
  * temporal drift: the ring share falls over time, the risky e-mail domains change at 60% of the
    period and amounts inflate, so a model trained early should decay later.
The signal is invented. Metrics on this data say nothing about real fraud.
"""
import numpy as np
import pandas as pd

T0 = 86400  # real TransactionDT starts around one day, not zero


def make_transactions(n=30000, seed=0, days=180, fraud_rate=0.035):
    """Return (transaction_df, identity_df), both in shuffled row order."""
    rng = np.random.default_rng(seed)
    span = days * 86400
    n_fraud = int(round(n * fraud_rate))
    n_legit = n - n_fraud

    # ---- card profiles (the stable account proxy) ----
    n_cards = max(50, n // 8)
    cards = pd.DataFrame({
        "card1": rng.integers(1000, 18000, n_cards),
        "card2": rng.choice([100, 111, 125, 321, 490, 514, 555], n_cards),
        "card3": rng.choice([150, 185, 144], n_cards, p=[0.85, 0.1, 0.05]),
        "card4": rng.choice(["visa", "mastercard", "american express", "discover"], n_cards, p=[0.65, 0.3, 0.03, 0.02]),
        "card5": rng.choice([226, 224, 166, 102], n_cards),
        "card6": rng.choice(["debit", "credit"], n_cards, p=[0.75, 0.25]),
        "addr1": rng.choice([204, 299, 325, 441, 264, 315], n_cards).astype(float),
        "addr2": 87.0,
        "mu": rng.normal(3.8, 0.7, n_cards),
        "email": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"], n_cards),
        "device": rng.choice(["Windows", "iOS Device", "MacOS", "SM-G960F Build/R16NW", "Moto G (5) Build/NPPS25"], n_cards),
    })
    weights = rng.pareto(2.0, n_cards) + 1
    weights /= weights.sum()

    # ---- legitimate rows ----
    legit_card = rng.choice(n_cards, n_legit, p=weights)
    legit_t = rng.uniform(0, span, n_legit)

    # ---- fraud rows: ring bursts + isolated events ----
    frows = []
    remaining = n_fraud
    while remaining > 0:
        t_start = rng.uniform(0, span - 7200)
        p_ring = 0.7 - 0.4 * (t_start / span)
        if rng.random() < p_ring:
            size = int(min(remaining, rng.integers(4, 9)))
            ring_cards = rng.choice(n_cards, 2, replace=False)
            ring_dev = f"RING-{rng.integers(0, 10 ** 6)} Build/X"
            for _ in range(size):
                frows.append((int(rng.choice(ring_cards)), t_start + rng.uniform(0, 7200), ring_dev))
            remaining -= size
        else:
            frows.append((int(rng.integers(0, n_cards)), t_start, None))
            remaining -= 1
    fr = pd.DataFrame(frows, columns=["c", "t", "ring_dev"])

    c_idx = np.concatenate([legit_card, fr["c"].to_numpy()])
    t = np.concatenate([legit_t, fr["t"].to_numpy()])
    f = np.concatenate([np.zeros(n_legit, bool), np.ones(len(fr), bool)])
    ring_dev = np.concatenate([np.full(n_legit, None, dtype=object), fr["ring_dev"].to_numpy(dtype=object)])
    m = len(t)
    frac = t / span
    prof = cards.iloc[c_idx].reset_index(drop=True)

    # ---- amount: card habit, global inflation, fraud multiplier ----
    amt = np.exp(prof["mu"].to_numpy() + rng.normal(0, 0.6, m)) * (1 + 0.3 * frac)
    amt = np.where(f, amt * rng.uniform(1.0, 2.6, m), amt)
    amt = np.round(np.clip(amt, 0.5, 5000), 2)

    prod = np.where(f, rng.choice(list("WCHRS"), m, p=[0.3, 0.45, 0.1, 0.1, 0.05]),
                    rng.choice(list("WCHRS"), m, p=[0.7, 0.1, 0.1, 0.07, 0.03]))

    # ---- e-mail domains: the risky set changes over time (concept drift) ----
    risky_pick = np.where(frac < 0.6, rng.choice(["anonymous.com", "protonmail.com"], m),
                          rng.choice(["anonymous.com", "mail.com"], m))
    use_risky = (f & (rng.random(m) < 0.5)) | ((~f) & (rng.random(m) < 0.03))
    p_email = np.where(use_risky, risky_pick, prof["email"].to_numpy()).astype(object)
    p_email[rng.random(m) < 0.1] = np.nan
    r_email = np.where(use_risky, risky_pick, prof["email"].to_numpy()).astype(object)
    r_email[rng.random(m) < 0.55] = np.nan

    # ---- C counts, D deltas, M flags, V columns ----
    lam = np.array([1.2, 1.2, 0.1, 0.1, 0.8, 1.0, 0.8, 0.1, 0.1, 0.7, 0.9, 1.0, 0.8, 1.5])
    mult = np.ones(14)
    mult[[0, 1, 5, 10, 13]] = 1.6
    C = rng.poisson(np.where(f[:, None], lam * mult, lam)).astype(float) + 1
    D = {}
    for j in range(1, 16):
        d = rng.exponential(25 if j in (1, 2) else 40, m)
        if j == 1:
            d = np.where(f, d * 0.5, d)
        d = np.round(np.clip(d, 0, 600))
        d[rng.random(m) < min(0.9, 0.1 + 0.05 * j)] = np.nan
        D[f"D{j}"] = d
    M = {}
    for j in range(1, 10):
        if j == 4:
            vals = rng.choice(["M0", "M1", "M2"], m, p=[0.5, 0.3, 0.2]).astype(object)
        else:
            pf = 0.3 if j in (2, 3, 6) else 0.18
            vals = np.where(rng.random(m) < np.where(f, pf, 0.15), "F", "T").astype(object)
        vals[rng.random(m) < 0.4] = np.nan
        M[f"M{j}"] = vals
    V = {}
    for j in range(1, 13):
        v = np.round(np.clip(rng.normal(1 + 0.5 * f * (j % 3 == 0), 1, m), 0, None))
        v[rng.random(m) < (0.45 if j > 6 else 0.02)] = np.nan
        V[f"V{j}"] = v

    dist1 = np.round(np.where(f, rng.exponential(110, m), rng.exponential(60, m)))
    dist1[rng.random(m) < 0.6] = np.nan

    df = pd.DataFrame({
        "isFraud": f.astype(int),
        "TransactionDT": (T0 + t).astype(int),
        "TransactionAmt": amt,
        "ProductCD": prod,
        "card1": prof["card1"], "card2": prof["card2"], "card3": prof["card3"],
        "card4": prof["card4"], "card5": prof["card5"], "card6": prof["card6"],
        "addr1": prof["addr1"], "addr2": prof["addr2"], "dist1": dist1,
        "P_emaildomain": p_email, "R_emaildomain": r_email,
    })
    for j in range(14):
        df[f"C{j + 1}"] = C[:, j]
    for block in (D, M, V):
        for k, v in block.items():
            df[k] = v
    df["_device"] = np.where(pd.notna(ring_dev), ring_dev, prof["device"].to_numpy())

    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    df.insert(0, "TransactionID", np.arange(2987000, 2987000 + m))

    # ---- identity table: only a share of transactions has one, fraud more often ----
    has_id = rng.random(m) < np.where(df["isFraud"] == 1, 0.55, 0.22)
    sub = df[has_id]
    k = len(sub)
    dev = sub["_device"].astype(str)
    ident = pd.DataFrame({
        "TransactionID": sub["TransactionID"].to_numpy(),
        "DeviceType": np.where(dev.str.contains("Build|iOS"), "mobile", "desktop"),
        "DeviceInfo": dev.to_numpy(),
        "id_01": np.round(-rng.exponential(10, k)),
        "id_02": np.round(rng.lognormal(11, 1, k)),
        "id_30": rng.choice(["Windows 10", "Windows 7", "iOS 11.2.1", "Android 7.0", "Mac OS X 10_12_6"], k),
        "id_31": rng.choice(["chrome 63.0", "safari 11.0", "mobile safari 11.0", "firefox 57.0", "edge 16.0"], k),
        "id_33": rng.choice(["1920x1080", "1366x768", "2208x1242", "1440x900"], k),
    })
    df = df.drop(columns="_device")
    # shuffle row order: nothing downstream may rely on the file being time-sorted
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    ident = ident.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df, ident
