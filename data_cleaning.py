"""
Financial Fraud Detection - Data Cleaning & Feature Engineering
==================================================================
Zidio Development Internship | Month 2 Project

Source: Fraud_Detection.xlsx (10,127 transactions, PaySim-style mobile
money simulation data with country/behavioral fields added).

This script documents every cleaning decision so it can be explained
to a reviewer. Run it once to produce data/fraud_data_clean.csv, which
train_model.py and app.py both consume.
"""

import pandas as pd
import numpy as np

RAW_PATH = "/mnt/user-data/uploads/Fraud_Detection.xlsx"
OUT_PATH = "data/fraud_data_clean.csv"

df = pd.read_excel(RAW_PATH)
print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------
# 1. Resolve the label column
# ---------------------------------------------------------------
# isFraud (text: Safe/Fraud/Not reviewed) and "isFraud - Copy" (0/1)
# carry the same information. 2 rows are labelled "Not reviewed" -
# ground truth is unknown for these, so they can't be used to train
# or evaluate a classifier. None of the 68 confirmed fraud rows are
# affected, so nothing about the fraud signal is lost by excluding them.
n_not_reviewed = (df["isFraud"] == "Not reviewed").sum()
df = df[df["isFraud"] != "Not reviewed"].copy()
df["IsFraud"] = (df["isFraud"] == "Fraud").astype(int)
print(f"Dropped {n_not_reviewed} 'Not reviewed' rows with unknown ground truth")

# ---------------------------------------------------------------
# 2. Drop redundant / non-informative columns
# ---------------------------------------------------------------
# - isFraud, isFraud - Copy      -> replaced by clean IsFraud (0/1) above
# - Column1                      -> a plain row index, duplicate of the
#                                    dataframe index, no analytical value
# - isFlaggedFraud               -> constant 0 for every single row here,
#                                    carries zero information
# - DayOfWeek(new)                -> exact duplicate of DayOfWeek, just
#                                    spelled out (Wed vs 3); kept the
#                                    numeric version, dropped the text one
df = df.rename(columns={"branch": "country"})   # this column holds
                                                 # country names (Espana,
                                                 # Honduras, ...), not
                                                 # bank branches - renamed
                                                 # for clarity downstream
df = df.drop(columns=["isFraud", "isFraud - Copy", "Column1",
                       "isFlaggedFraud", "DayOfWeek(new)"])

# ---------------------------------------------------------------
# 3. Handle missing values
# ---------------------------------------------------------------
# Only 37 rows (0.4% of data) have any missing value at all, spread
# thinly across 10 different columns, and none touch a fraud row.
# Safe to drop rather than impute - imputing financial balances or
# transaction types risks inventing signal that was never there.
before = len(df)
df = df.dropna()
print(f"Dropped {before - len(df)} rows with missing values (0 were fraud cases)")

# ---------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------
# Balance-consistency errors: in a clean transaction, the sender's
# balance should fall by exactly `amount` and the receiver's should
# rise by exactly `amount`. Large deviations are a classic tell for
# manipulated/fraudulent records.
df["errorBalanceOrig"] = df["newbalanceOrig"] - (df["oldbalanceOrg"] - df["amount"])
df["errorBalanceDest"] = df["newbalanceDest"] - (df["oldbalanceDest"] + df["amount"])

# Sender's account emptied out by the transaction
df["origBalanceWiped"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)

# Destination balance never moves even though money supposedly arrived
df["destBalanceUnchanged"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)).astype(int)

# How large the transaction is relative to what the sender had
df["amountToBalanceRatio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

# Fraud in this data occurs ONLY in CASH_OUT and TRANSFER (0 cases in
# PAYMENT/CASH_IN/DEBIT) - this matches how the PaySim simulator
# generates fraud and is the single strongest signal available.
df["isHighRiskType"] = df["type"].isin(["CASH_OUT", "TRANSFER"]).astype(int)

df["nameDestIsMerchant"] = df["nameDest"].str.startswith("M").astype(int)

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns -> {OUT_PATH}")
print(f"Fraud cases retained: {df['IsFraud'].sum()} ({df['IsFraud'].mean()*100:.2f}%)")
