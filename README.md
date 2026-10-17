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
- **Interpretability:** SHAP summary (computed on the model set in `modeling.shap_model`, currently `lightgbm_spw`, not on the best table row `lightgbm`).
- **Simple label-delay model:** fraud labels arrive late (chargebacks). The gap between train end and test start stands in for that delay so train labels are mature when scoring starts. This is a documented simplification, not a chargeback-timing model.

### Results

Real IEEE-CIS data: 590,540 transactions over about 182 days, 3.5% fraud. Strict time split: **train** = earliest 413,378 transactions (3.52% fraud), **test** = latest 154,030 (3.47% fraud), with a gap between them. One run, one seed; no confidence intervals. Alert budget = share of test transactions a team could review. Full tables and plots: `docs/real_run/`.

| Model | PR-AUC | Recall @0.5% | Recall @1% | Recall @2% | Precision @1% | Brier | ECE |
|-------|--------|--------------|------------|------------|---------------|-------|-----|
| rule (hand-set, no labels) | 0.044 | 0.7% | 0.8% | 1.7% | 2.8% | n/a | n/a |
| logistic | 0.417 | 13.1% | 22.5% | 33.6% | 78.2% | 0.026 | 0.007 |
| logistic_balanced | 0.404 | 12.9% | 21.9% | 31.7% | 76.0% | 0.149 | 0.284 |
| **lightgbm** | **0.500** | 13.1% | **24.3%** | **38.2%** | **84.2%** | **0.023** | **0.002** |
| lightgbm_spw | 0.479 | 13.1% | 24.3% | 36.8% | 84.4% | 0.071 | 0.156 |

Test prevalence is 3.47%, so precision of 84% at a 1% budget is a lift of about 24 over random review.

What the run shows:

- **The hand-set rule is close to useless** (PR-AUC 0.044 against a 0.035 prevalence), so the learned models are earning their keep.
- **LightGBM is best on every metric** (PR-AUC 0.50 vs 0.42 for logistic regression). At a 1% review budget it catches 24% of all fraud with 84% of alerts correct; at 2% it catches 38%.
- **Class weights hurt calibration, not ranking.** `logistic_balanced` and `lightgbm_spw` rank about as well as their unweighted versions but their probabilities are badly off (ECE 0.28 and 0.16 vs 0.007 and 0.002). If scores are used as probabilities in a cost decision, do not use class weighting.
- **Recall is small in absolute terms** because the budget is small: with 1% reviewed, at most about 29% of fraud (1% of transactions / 3.47% prevalence) could be caught even with perfect alerts. Always read recall together with the budget.
- **Decay inside the test window (about 55 days, four ~14-day buckets) is not monotonic.** LightGBM's per-bucket PR-AUC is 0.55, 0.51, 0.44, 0.52 and its recall at 1% is 24%, 26%, 25%, 21%. Score PSI against train stays below 0.03. With only four buckets and one split, I would not read this as proof of stability or decay.
- **Cost view (assumed costs, not from any institution):** expected cost with no review is 820,350; LightGBM reaches 315,173, the rule 638,517. The minimum for every model sits at the largest budget evaluated (10%), so the cost-optimal budget under these assumptions is at least that large; the grid should be widened before reading it as a recommendation.
- **Top SHAP features** are `M4`, `C13`, `TransactionAmt`, `C5`, `C1`, `C14`, `card6`, `D2`. Most are anonymised columns, so the explanation says little about real-world causes.

### Known issues found by an independent review

An independent, AI-assisted review of this repo (read-only; the numbers below were recomputed on the real data by the reviewer) changes how the headline should be read:

- **The advertised point-in-time card/device features add almost nothing.** Dropping all of them: PR-AUC 0.502 vs 0.500. Dropping the Vesta columns `C1-C14` and `D1-D15` instead: PR-AUC 0.302 and precision at a 1% budget falls from 84% to 55%. The skill comes from those engineered columns, not from the behavioural features this project builds.
- **The test set rewards recognising repeat victims.** 74% of test frauds sit on a card key that already had a fraud in training, and 74% of the top-1% alerts are `ProductCD = C` (10% of rows). On the subset where the card has no fraud in training, precision at the 0.5% budget is about 76% instead of 90.8%.
- **Four different model families pick almost the same 699 frauds in the top 0.5%**, which points to one easily separable cluster of attacked cards. A card-cluster bootstrap gives a wider interval for precision at 0.5% (about 86-94%) than a row bootstrap.
- **Label delay is not modelled.** The 7-day gap does not remove labels that would not yet exist at deployment. Training only on labels older than 30 days before the cut-off lowers PR-AUC from 0.50 to about 0.455.
- **The cost curve's grid stops at 10% of transactions**, and the minimum sits at that edge, so no optimal budget is established.
- **The card key** (built from `card1-6` and `addr1`) is an unvalidated proxy; some keys cover thousands of rows and are not one cardholder.
- **One split, one seed, hand-set hyperparameters,** no confidence intervals in the table above.

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
- Tests and CI run on **synthetic data with an invented signal** (fraud rings, drift injected on purpose). Synthetic metrics say nothing about real performance. The real-data results above come from one run on the Kaggle files, which are not in this repo.
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

რეალური IEEE-CIS მონაცემები: 590 540 ტრანზაქცია დაახლოებით 182 დღეზე, 3.5% თაღლითობა. მკაცრი დროითი გაყოფა: **train** = უადრესი 413 378 ტრანზაქცია (3.52%), **test** = უახლესი 154 030 (3.47%), მათ შორის დროის შუალედით. ერთი გაშვება, ერთი seed; სანდოობის ინტერვალები არ არის. სრული ცხრილები და გრაფიკები: `docs/real_run/`.

| მოდელი | PR-AUC | Recall @0.5% | Recall @1% | Recall @2% | Precision @1% | Brier | ECE |
|--------|--------|--------------|------------|------------|---------------|-------|-----|
| წესი (ხელით, ლეიბლების გარეშე) | 0.044 | 0.7% | 0.8% | 1.7% | 2.8% | n/a | n/a |
| logistic | 0.417 | 13.1% | 22.5% | 33.6% | 78.2% | 0.026 | 0.007 |
| logistic_balanced | 0.404 | 12.9% | 21.9% | 31.7% | 76.0% | 0.149 | 0.284 |
| **lightgbm** | **0.500** | 13.1% | **24.3%** | **38.2%** | **84.2%** | **0.023** | **0.002** |
| lightgbm_spw | 0.479 | 13.1% | 24.3% | 36.8% | 84.4% | 0.071 | 0.156 |

- LightGBM ყველა მეტრიკით საუკეთესოა. 1%-იანი შემოწმების ბიუჯეტით ის თაღლითობის 24%-ს იჭერს, ხოლო შეტყობინებების 84% სწორია; 2%-ით იჭერს 38%-ს.
- კლასის წონები კალიბრაციას აზიანებს და არა დახარისხებას: `logistic_balanced`-ის ECE 0.28, ხოლო `lightgbm_spw`-ის 0.16, ანუ ალბათობები არასწორია. თუ ქულა ალბათობად გამოიყენება, წონები არ გამოიყენოთ.
- ხელით დაწერილი წესი თითქმის უსარგებლოა (PR-AUC 0.044, პრევალენტობა 0.035).
- ტესტის დაახლოებით 55 დღის განმავლობაში დაცემა მონოტონური არ არის (LightGBM PR-AUC ბაკეტებში: 0.55, 0.51, 0.44, 0.52); მხოლოდ ერთი გაყოფის გამო სტაბილურობას ან დაცემას ამით ვერ დავამტკიცებთ.
- ხარჯები დაშვებებია. მინიმალური ხარჯი ყველა მოდელზე ყველაზე დიდ შემოწმებულ ბიუჯეტზეა (10%), ამიტომ ბადე გასაფართოებელია, სანამ ეს რეკომენდაციად გამოიყენება.

### გაშვება

```bash
pip install -r requirements.txt
pytest
python run_experiment.py --synthetic
python run_experiment.py
```

რეალური მონაცემებისთვის ჩამოტვირთეთ Kaggle-დან `train_transaction.csv` და `train_identity.csv` დირექტორიაში `data/raw/`.

### ცნობილი ხარვეზები დამოუკიდებელი შემოწმებიდან

- წარმოდგენილი point-in-time ბარათის/მოწყობილობის ნიშნები თითქმის არაფერს სძენს (PR-AUC 0.502 მათ გარეშე vs 0.500); უნარი C1-C14 და D1-D15 სვეტებიდან მოდის (მათ გარეშე PR-AUC 0.302).
- ტესტში თაღლითობის 74% ისეთ ბარათზეა, რომელსაც train-ში უკვე ჰქონდა თაღლითობა; ტოპ 1% გაფრთხილების 74% `ProductCD = C`-ზეა.
- ლეიბლის დაგვიანება მოდელირებული არ არის (30 დღიანი დაგვიანებით PR-AUC 0.50-დან დაახლოებით 0.455-მდე ეცემა).
- ტესტის ფანჯარა დაახლოებით 55 დღეა და არა 4 თვე. ხარჯის მრუდის ბადე 10%-ზე ჩერდება. ერთი გაყოფა, ერთი seed.

### შეზღუდვები

- **`TransactionDT` ფარდობითი წამებია** უცნობი საწყისი თარიღიდან. თანმიმდევრობა სანდოა, კალენდარული ეფექტები (კვირის დღე, არდადეგები) არა.
- **ბარათის გასაღები პროქსია.** IEEE-CIS-ში მომხმარებლის ID არ არის; გასაღები `card1–card6`-სა და `addr1`-ისგან შედგება, ამიტომ შეიძლება სხვადასხვა ადამიანი გააერთიანოს ან ერთი ადამიანი გაყოს.
- **V ნიშნები ანონიმიზებულია;** SHAP აჩვენებს რომ მნიშვნელოვანია, მაგრამ არა რატომ.
- **ლეიბლის დაგვიანება გამარტივებულია:** რეალური chargeback-ები კვირების განმავლობაში მოდის; ფაილის ბოლოს ლეიბლები შეიძლება ჯერ არასრული იყოს.
- **ხარჯები დაშვებებია,** მონაცემებიდან შეფასებული არა.
- დალაგება მხოლოდ ქულით ხდება; პრაქტიკაში შეიძლება მოსალოდნელი ზარალით (ალბათობა × თანხა).
- ტესტები და CI **სინთეზურ მონაცემებზე გამოგონილი სიგნალით** მუშაობს (თაღლითთა ჯგუფები და დრიფტი განზრახ არის ჩადებული), ამიტომ ეს მეტრიკები რეალურ შედეგს არ ასახავს. რეალურ მონაცემებზე გაშვება ჯერ არ შესრულებულა.
- ~3.5% თაღლითობა შეჯიბრის ნიმუშის მაჩვენებელია; რეალურ სისტემაში ის, როგორც წესი, გაცილებით დაბალია.
