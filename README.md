<p align="center">
  <a href="#english">🇺🇸 English</a> &nbsp;•&nbsp;
  <a href="#georgian">🇬🇪 ქართული</a>
</p>

<a id="english"></a>

<h1 align="center">Card Fraud Detection</h1>
<p align="center"><em>Strict time split · point-in-time behavioural features · alert budgets · expected cost · drift</em></p>

---

### Problem and decision

A bank or payment fintech has a fraud-review team that can look at only a small share of transactions (say 0.5–2%). The model's job is to **rank** transactions so that the reviewed slice contains as much fraud as possible. The questions this project answers are therefore not "what is the AUC" but:

- How much fraud is caught at a fixed **alert budget** (recall and precision at the top 0.5% / 1% / 2%)?
- At what budget does reviewing stop paying for itself, under **stated, assumed costs**?
- Does the model still work **later in time**, and does the score distribution drift?

### Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) on Kaggle: `train_transaction.csv` (~590k card transactions, ~3.5% fraud) and `train_identity.csv` (device / browser info for a minority of transactions). **The data cannot be redistributed from this repo.** Download it yourself and check the competition rules and licence/terms before using it; place the files in `data/raw/`.

**Why not PaySim?** PaySim is itself a simulation, so a model on it learns the simulator's rules. IEEE-CIS is real (Vesta) card traffic, which makes drift, anonymised features and weak labels real problems instead of toy ones.

### What is in the project

- **Time-ordered split, never random.** Train = earliest 70% of transactions by time, then a gap (default 7 days), test = everything later. A test asserts that no test transaction precedes any train transaction, and the pipeline raises if it does.
- **Point-in-time features** per card proxy: expanding transaction count, expanding mean/std of amount, seconds since the previous transaction, amount z-score against that card's own *past*, and velocity counts in the last 1 h and 24 h; plus the same counts per device. Each row uses only strictly earlier rows (ordered by time, then `TransactionID`). Labels are never used in features. A leakage test removes or alters later rows and proves earlier rows' features do not change.
- **Models:** hand-set rule baseline (amount z-score + 1 h velocity + unseen card; not fitted on labels), logistic regression and LightGBM, each with and without imbalance handling (`class_weight="balanced"` / `scale_pos_weight`). **No SMOTE.** Weighting changes predicted probabilities, so Brier score, ECE and calibration curves are reported next to ranking metrics.
- **Capacity-aware evaluation:** PR-AUC, recall and precision at fixed budgets (`budget_table.csv`), expected-cost curve (`cost_curve.csv/.png`) with costs set in `configs/config.yaml` (review cost per alert vs loss on missed fraud, either amount-proportional or fixed). Ties in scores are broken randomly, not by row order.
- **Decay and drift:** per-time-bucket PR-AUC and recall inside the test period (`bucket_performance.csv`), PSI of scores and key features versus train (`drift.csv`). Train scores are in-sample, so PSI is also given against the first test bucket.
- **Interpretability:** SHAP summary for LightGBM.
- **Simple label-delay model:** fraud labels arrive late (chargebacks). The gap between train end and test start stands in for that delay so train labels are mature when scoring starts. This is a documented simplification, not a chargeback-timing model.

### Results

Run on the real data to fill these tables (no real-data results exist yet; numbers produced on synthetic data are not reported here on purpose).

| Model | PR-AUC | Recall @0.5% | Recall @1% | Recall @2% | Precision @1% | Brier | ECE |
|-------|--------|--------------|------------|------------|---------------|-------|-----|
| rule | _run on real data_ | | | | | n/a | n/a |
| logistic | | | | | | | |
| logistic_balanced | | | | | | | |
| lightgbm | | | | | | | |
| lightgbm_spw | | | | | | | |

| Cost view (assumed costs) | Value |
|---------------------------|-------|
| Cost with no review | _run on real data_ |
| Minimum expected cost / at budget | |
| Recall by test bucket (decay) | |
| Score PSI, last test bucket vs train | |

### How to run

```bash
pip install -r requirements.txt
pytest                                      # all tests use synthetic data
python run_experiment.py --synthetic        # end-to-end smoke test on synthetic data
python run_experiment.py                    # real data from configs/config.yaml (data/raw/)
python run_experiment.py --rows 100000      # earliest 100k rows only (never a random sample)
```

Outputs go to `reports/`: `metrics.csv`, `budget_table.csv`, `cost_curve.csv`, `drift.csv`, `bucket_performance.csv`, `shap_importance.csv` and plots (`pr_curves`, `budget_recall`, `cost_curve`, `calibration`, `decay_recall`, `score_psi`, `shap_summary`).

### Limitations

- **`TransactionDT` is a relative offset in seconds** from an undisclosed reference date. Ordering is reliable; calendar effects (weekday, holidays) are not, and `hour` is only a guess at time of day.
- **The card key is a proxy.** IEEE-CIS has no customer ID; the key is built from `card1–card6` and `addr1`, so it can merge different people or split one person. Behavioural features are only as good as that proxy.
- **V features are anonymised** engineered columns; SHAP can say they matter but not why.
- **Label delay is simplified.** Real chargebacks arrive over weeks and are censored near the end of the data; here only a fixed gap models that. Labels in the last days before the end of the file may still be immature.
- **Costs are assumptions**, not estimates from data. Amount-proportional loss ignores recoveries, customer friction and fees.
- Ranking is by score only; a production system might rank by expected loss (probability × amount).
- Tests and CI run on **synthetic data with an invented signal** (fraud rings, drift injected on purpose). Synthetic metrics say nothing about real performance; the real-data run has not been done.
- The ~3.5% fraud rate is the competition sample; real fraud is usually far rarer, so budgets and precision will differ in production.

### Structure

```
├── run_experiment.py          # features, time split, models, evaluation, drift, SHAP -> reports/
├── configs/config.yaml        # paths, split, features, model params, alert budgets, assumed costs
├── src/fraud_detection/
│   ├── data.py                # load and join transaction + identity tables
│   ├── split.py               # strict time split + guard
│   ├── features.py            # point-in-time history features, preprocessors
│   ├── models.py              # rule baseline, logistic, LightGBM (+ imbalance variants)
│   ├── evaluate.py            # budgets, cost curve, calibration, PSI, per-bucket decay
│   ├── explain.py             # SHAP
│   ├── plots.py
│   └── synthetic.py           # IEEE-CIS-layout test data with rings and drift
└── tests/                     # pytest + GitHub Actions
```

<p align="right"><a href="#georgian">🇬🇪 ქართული ↓</a></p>

---

<a id="georgian"></a>

<h1 align="center">ბარათით თაღლითობის აღმოჩენა</h1>
<p align="center"><em>მკაცრი დროითი გაყოფა · წერტილოვანი (point-in-time) ქცევითი ნიშნები · შემოწმების ბიუჯეტი · მოსალოდნელი ხარჯი · დრიფტი</em></p>

### ამოცანა და გადაწყვეტილება

ბანკს ან გადახდების ფინტექ კომპანიას თაღლითობის შემმოწმებელი გუნდი ჰყავს, რომელსაც ტრანზაქციების მხოლოდ მცირე ნაწილის (მაგ. 0.5–2%) გადახედვა შეუძლია. მოდელის ამოცანაა ტრანზაქციების **დალაგება** ისე, რომ შესამოწმებელ ნაწილში რაც შეიძლება მეტი თაღლითობა მოხვდეს. ამიტომ პროექტი არა „რა არის AUC“-ს, არამედ შემდეგ კითხვებს პასუხობს:

- რამდენ თაღლითობას ვიჭერთ ფიქსირებული **ბიუჯეტის** პირობებში (recall და precision ტოპ 0.5% / 1% / 2%-ზე)?
- რა ბიუჯეტის შემდეგ აღარ ამართლებს შემოწმება თავს, **დაშვებული ხარჯების** პირობებში?
- მუშაობს თუ არა მოდელი **უფრო გვიან პერიოდში** და იცვლება თუ არა ქულების განაწილება?

### მონაცემები

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle): `train_transaction.csv` (~590 ათასი ტრანზაქცია, ~3.5% თაღლითობა) და `train_identity.csv` (მოწყობილობის/ბრაუზერის ინფორმაცია ტრანზაქციების უმცირესობისთვის). **მონაცემების გავრცელება ამ რეპოზიტორიიდან შეუძლებელია.** ჩამოტვირთეთ თავად, გადაამოწმეთ შეჯიბრის წესები და ლიცენზია/პირობები და მოათავსეთ `data/raw/`-ში.

**რატომ არა PaySim?** PaySim თავად სიმულაციაა, ამიტომ მასზე მოდელი სიმულატორის წესებს სწავლობს. IEEE-CIS რეალური ბარათის ტრაფიკია, ამიტომ დრიფტი, ანონიმიზებული ნიშნები და არასრულყოფილი ლეიბლები რეალური პრობლემებია.

### რა შედის პროექტში

- **დროით დალაგებული გაყოფა, არასდროს შემთხვევითი.** Train = დროით პირველი 70%, შემდეგ დროის ხვრელი (სტანდარტულად 7 დღე), test = ყველაფერი ამის შემდეგ. ტესტი ამოწმებს, რომ არცერთი test ტრანზაქცია არ უსწრებს train-ის რომელიმეს; წინააღმდეგ შემთხვევაში პაიპლაინი ჩერდება.
- **Point-in-time ნიშნები** ბარათის პროქსისთვის: ტრანზაქციების რაოდენობა, თანხის საშუალო/სტანდარტული გადახრა, წინა ტრანზაქციიდან გასული წამები, თანხის z-ქულა ამავე ბარათის *წარსულთან* მიმართებით, სიჩქარე (ბოლო 1 და 24 საათი) და იგივე მოწყობილობაზე. თითოეული სტრიქონი იყენებს მხოლოდ მკაცრად უფრო ადრინდელ სტრიქონებს; ლეიბლები ნიშნებში არასდროს გამოიყენება. ტესტი შემდგომ სტრიქონებს შლის ან ცვლის და ამტკიცებს, რომ ადრინდელი სტრიქონების ნიშნები არ იცვლება.
- **მოდელები:** ხელით დაწერილი წესების baseline (ლეიბლებზე არ არის დასწავლილი), ლოგისტიკური რეგრესია და LightGBM, თითოეული დისბალანსის დამუშავებით და მის გარეშე (`class_weight="balanced"` / `scale_pos_weight`). **SMOTE არ გამოიყენება.** წონები ალბათობებს ცვლის, ამიტომ Brier, ECE და კალიბრაციის მრუდები ranking მეტრიკების გვერდით ჩანს.
- **ბიუჯეტზე დაფუძნებული შეფასება:** PR-AUC, recall/precision ფიქსირებულ ბიუჯეტზე, მოსალოდნელი ხარჯის მრუდი `configs/config.yaml`-ში მითითებული ხარჯებით (შემოწმების ხარჯი vs გამოტოვებული თაღლითობის ზარალი). თანაბარი ქულები შემთხვევითად იშლება და არა სტრიქონის რიგით.
- **დაცემა და დრიფტი:** PR-AUC და recall test პერიოდის დროით ინტერვალებში, ქულებისა და ძირითადი ნიშნების PSI train-თან შედარებით. Train ქულები in-sample-ია, ამიტომ PSI პირველ test ინტერვალთანაც არის მოცემული.
- **ინტერპრეტაცია:** SHAP LightGBM-ისთვის.
- **ლეიბლის დაგვიანების მარტივი მოდელი:** თაღლითობის ლეიბლები (chargeback) გვიან მოდის; train-ის დასასრულსა და test-ის დასაწყისს შორის ხვრელი ამ დაგვიანებას ცვლის. ეს გამარტივებაა და არა chargeback-ის დროის მოდელი.

### შედეგები

რეალურ მონაცემებზე გაშვების შემდეგ შეავსეთ ცხრილები (რეალური შედეგები ჯერ არ არსებობს; სინთეზურ მონაცემებზე მიღებული რიცხვები განზრახ არ არის მოყვანილი). ცხრილები იხილეთ ინგლისურ ნაწილში.

### გაშვება

```bash
pip install -r requirements.txt
pytest
python run_experiment.py --synthetic
python run_experiment.py
```

რეალური მონაცემებისთვის ჩამოტვირთეთ Kaggle-დან `train_transaction.csv` და `train_identity.csv` დირექტორიაში `data/raw/`.

### შეზღუდვები

- **`TransactionDT` ფარდობითი წამებია** უცნობი საწყისი თარიღიდან. თანმიმდევრობა სანდოა, კალენდარული ეფექტები (კვირის დღე, არდადეგები) არა.
- **ბარათის გასაღები პროქსია.** IEEE-CIS-ში მომხმარებლის ID არ არის; გასაღები `card1–card6`-სა და `addr1`-ისგან შედგება, ამიტომ შეიძლება სხვადასხვა ადამიანი გააერთიანოს ან ერთი ადამიანი გაყოს.
- **V ნიშნები ანონიმიზებულია;** SHAP აჩვენებს რომ მნიშვნელოვანია, მაგრამ არა რატომ.
- **ლეიბლის დაგვიანება გამარტივებულია:** რეალური chargeback-ები კვირების განმავლობაში მოდის; ფაილის ბოლოს ლეიბლები შეიძლება ჯერ არასრული იყოს.
- **ხარჯები დაშვებებია,** მონაცემებიდან შეფასებული არა.
- დალაგება მხოლოდ ქულით ხდება; პრაქტიკაში შეიძლება მოსალოდნელი ზარალით (ალბათობა × თანხა).
- ტესტები და CI **სინთეზურ მონაცემებზე გამოგონილი სიგნალით** მუშაობს (თაღლითთა ჯგუფები და დრიფტი განზრახ არის ჩადებული), ამიტომ ეს მეტრიკები რეალურ შედეგს არ ასახავს. რეალურ მონაცემებზე გაშვება ჯერ არ შესრულებულა.
- ~3.5% თაღლითობა შეჯიბრის ნიმუშის მაჩვენებელია; რეალურ სისტემაში ის, როგორც წესი, გაცილებით დაბალია.
