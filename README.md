# Financial Fraud Detection Model — Project Notes

## What this actually is
The brief your team shared described a full production system (Kafka/Spark
streaming, autoencoders, blockchain, mobile alerts). What you were actually
assigned is the **Data Analytics deliverable**: *"Financial Fraud Detection
Model (Data Analytics – Streamlit or Power BI dashboard)."* This project
builds that — a cleaned dataset, a trained classifier, and an interactive
Streamlit dashboard — without the production infrastructure that's out of
scope for a monthly analytics project.

## Dataset
`Fraud_Detection.xlsx` — 10,127 mobile-money-style transactions (PaySim
schema: `step`, `type`, `amount`, origin/destination balances) with extra
fields layered on: `country` (mislabelled "branch" in the raw file),
`unusuallogin`, `Acct type`, `Time of day`, `DayOfWeek`. 68 confirmed fraud
cases (0.67%).

## Pipeline
1. **`data_cleaning.py`** — resolves the label, drops 2 rows with unknown
   ground truth and 37 rows with missing values (none of them fraud),
   removes redundant/constant columns, engineers balance-consistency
   features. See `data/fraud_data_clean.csv` (10,088 rows).
2. **`train_model.py`** — fraud only occurs in `CASH_OUT`/`TRANSFER`, so the
   classifier trains on that subset only (2,269 rows, 3.0% fraud) rather
   than the full 5-type dataset, which would let it "cheat" on transaction
   type alone. Compares Logistic Regression, Random Forest, and XGBoost on
   precision/recall/F1/ROC-AUC/PR-AUC — not accuracy, which is meaningless
   under this much class imbalance.
3. **`app.py`** — four-tab Streamlit dashboard: Overview, Fraud Patterns,
   Model Performance, and a live Risk Scorer.

## Key findings worth mentioning Monday
- Fraud is confined entirely to `CASH_OUT` and `TRANSFER` — zero cases in
  `PAYMENT`, `CASH_IN`, `DEBIT` across 8,141 transactions.
- The best model (see the app's Model Performance tab) scores near-perfect
  on the held-out test set. **Flag this as a dataset limitation, not a
  result to oversell** — the engineered balance-consistency features
  nearly perfectly separate classes in this simulated data; real-world
  transaction data won't be this clean.
- `unusuallogin` is *lower* on average for fraud cases than safe ones —
  counter to the intuitive assumption, and worth a callout in review.

## Running it locally
```bash
pip install -r requirements.txt
python data_cleaning.py     # regenerates data/fraud_data_clean.csv
python train_model.py       # regenerates models/*.pkl and metrics.json
streamlit run app.py
```

## Deploying to Streamlit Cloud
- Set the Python version to **3.11** in Advanced Settings before deploying.
- Make sure `data/fraud_data_clean.csv` and everything in `models/` are
  actually committed and pushed to GitHub — a `.gitignore` or a
  size/extension rule silently excluding them is the most common cause of
  a working local app that breaks once deployed.
- `app.py` must sit at the repo root (or set it as the main file path in
  the Streamlit Cloud deploy settings).

## What's genuinely out of scope (and why that's fine)
Kafka/Spark real-time streaming, live email alerting, and graph-based
entity-network fraud detection are production-engineering concerns, not
data-analytics-dashboard concerns — they'd need infrastructure (a running
Kafka cluster, a mail server) this project has no reason to depend on. If
asked, the honest answer is that this dashboard analyzes and scores
transactions on demand; a production version would wire the same model
into a streaming pipeline.
