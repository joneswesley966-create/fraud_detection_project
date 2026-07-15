import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .kpi-card {
        background-color: #1a1f2b;
        border: 1px solid #2d3444;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: left;
    }
    .kpi-label { color: #9aa4b2; font-size: 0.85rem; margin-bottom: 4px; }
    .kpi-value { color: #f5f6f8; font-size: 1.7rem; font-weight: 700; }
    .kpi-sub { color: #6ee7a7; font-size: 0.8rem; }
    .kpi-sub-red { color: #f87171; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

DAY_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
             5: "Friday", 6: "Saturday", 7: "Sunday"}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HIGH_RISK_TYPES = ["CASH_OUT", "TRANSFER"]


@st.cache_data
def load_data():
    df = pd.read_csv("data/fraud_data_clean.csv")
    df["DayName"] = df["DayOfWeek"].map(DAY_NAMES)
    df["Status"] = df["IsFraud"].map({0: "Safe", 1: "Fraud"})
    return df


@st.cache_resource
def load_model():
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    with open("models/best_model_name.txt") as f:
        best_name = f.read().strip()
    return model, scaler, features, metrics, best_name


df = load_data()
model, scaler, FEATURES, metrics, BEST_NAME = load_model()

st.title("🛡️ Financial Fraud Detection Dashboard")
st.caption("Zidio Development Internship — Data Analytics Project · Data source: transaction-level banking simulation")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Fraud Patterns", "🤖 Model Performance", "⚡ Risk Scorer"])

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    total_txn = len(df)
    fraud_count = int(df["IsFraud"].sum())
    fraud_rate = df["IsFraud"].mean() * 100
    amount_at_risk = df.loc[df["IsFraud"] == 1, "amount"].sum()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Total Transactions", f"{total_txn:,}", "10,127 raw → cleaned"),
        (c2, "Confirmed Fraud Cases", f"{fraud_count}", f"{fraud_rate:.2f}% of all transactions"),
        (c3, "Amount at Risk", f"{amount_at_risk:,.0f}", "sum of fraudulent transactions"),
        (c4, "Fraud-Free Types", "3 of 5", "PAYMENT · CASH_IN · DEBIT"),
    ]:
        col.markdown(f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>""",
                      unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)
    with left:
        by_type = df.groupby("type").agg(transactions=("IsFraud", "count"),
                                          fraud=("IsFraud", "sum")).reset_index()
        fig = px.bar(by_type, x="type", y="transactions", color="fraud",
                     color_continuous_scale=["#334155", "#ef4444"],
                     title="Transaction Volume by Type (colored by fraud count)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        vol = df.groupby("country").size().reset_index(name="transactions").sort_values(
            "transactions", ascending=False).head(10)
        fig = px.bar(vol, x="transactions", y="country", orientation="h",
                     title="Top 10 Countries by Transaction Volume")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Data cleaning summary"):
        st.markdown("""
        - **Raw file:** 10,127 rows × 20 columns
        - **2 rows** labelled `"Not reviewed"` dropped — ground truth unknown, none were confirmed fraud
        - **37 rows** dropped for missing values (0.4% of data) — none were fraud cases
        - **Redundant columns removed:** `isFraud - Copy` (duplicate label), `Column1` (row index),
          `isFlaggedFraud` (constant 0, no signal), `DayOfWeek(new)` (text duplicate of `DayOfWeek`)
        - **`branch` renamed to `country`** — the column holds country names, not bank branches
        - **Result:** 10,088 clean rows, all 68 original fraud cases retained
        """)

# =========================================================
# TAB 2 — FRAUD PATTERNS
# =========================================================
with tab2:
    st.subheader("Where fraud actually happens")
    st.info("Every confirmed fraud case in this dataset is a **CASH_OUT** or **TRANSFER** transaction. "
            "PAYMENT, CASH_IN and DEBIT have zero fraud across 8,141 transactions.")

    col1, col2 = st.columns(2)
    with col1:
        rate_by_type = df.groupby("type")["IsFraud"].mean().mul(100).reset_index(name="fraud_rate")
        fig = px.bar(rate_by_type, x="type", y="fraud_rate",
                     title="Fraud Rate by Transaction Type (%)", color="fraud_rate",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fraud_df = df[df["IsFraud"] == 1]
        top_countries = fraud_df.groupby("country").agg(
            fraud_amount=("amount", "sum"), fraud_count=("IsFraud", "count")
        ).reset_index().sort_values("fraud_amount", ascending=False).head(10)
        fig = px.bar(top_countries, x="fraud_amount", y="country", orientation="h",
                     title="Top 10 Countries by Fraud Amount", color="fraud_count",
                     color_continuous_scale="Reds")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        by_day = df.groupby("DayName")["IsFraud"].sum().reindex(DAY_ORDER).reset_index()
        fig = px.bar(by_day, x="DayName", y="IsFraud", title="Fraud Cases by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        by_tod = df.groupby("Time of day")["IsFraud"].sum().reset_index()
        fig = px.bar(by_tod, x="Time of day", y="IsFraud", title="Fraud Cases by Time of Day")
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    fig = px.box(df, x="Status", y="amount", color="Status",
                 color_discrete_map={"Safe": "#334155", "Fraud": "#ef4444"},
                 log_y=True, title="Transaction Amount: Fraud vs Safe (log scale)")
    st.plotly_chart(fig, use_container_width=True)

    avg_login_fraud = df.loc[df["IsFraud"] == 1, "unusuallogin"].mean()
    avg_login_safe = df.loc[df["IsFraud"] == 0, "unusuallogin"].mean()
    st.warning(f"**Counter-intuitive finding:** average 'unusual login' count is actually *lower* for "
               f"fraud cases ({avg_login_fraud:.1f}) than safe ones ({avg_login_safe:.1f}) in this dataset. "
               f"It's not a reliable standalone fraud signal here — worth flagging rather than assuming "
               f"more logins automatically means more risk.")

# =========================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================
with tab3:
    st.subheader("Model comparison")
    st.caption("Trained only on CASH_OUT + TRANSFER transactions (2,269 rows, 3.0% fraud) — "
               "the only types where fraud occurs — so the model can't just memorize transaction type.")

    results_df = pd.DataFrame(metrics["results"]).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(results_df.style.highlight_max(axis=0, subset=[c for c in results_df.columns if c != "Model"],
                                                 color="#14532d"), use_container_width=True)

    st.error(f"**Read this before presenting it:** {BEST_NAME} scores a perfect 1.0 on every metric. "
             "That's a signal, not a trophy — it means the engineered balance-consistency features "
             "(`errorBalanceOrig`, `amountToBalanceRatio`, `origBalanceWiped`) almost perfectly separate "
             "the classes in this **simulated** dataset, on a small 454-row test set with only 14 fraud "
             "cases. Real transaction data is noisier; call this out as a dataset limitation in your "
             "write-up rather than claiming a production-ready 100% detector.")

    col1, col2 = st.columns(2)
    with col1:
        cm = metrics["confusion_matrix"]
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted Safe", "Predicted Fraud"], y=["Actual Safe", "Actual Fraud"],
            colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False))
        fig.update_layout(title=f"Confusion Matrix — {BEST_NAME}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fi = pd.DataFrame(list(metrics["feature_importance"].items()),
                           columns=["feature", "importance"]).head(10)
        fig = px.bar(fi, x="importance", y="feature", orientation="h",
                     title=f"Top Features — {BEST_NAME}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        roc = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=BEST_NAME))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                  line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        pr = metrics["pr_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=BEST_NAME))
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4 — RISK SCORER
# =========================================================
with tab4:
    st.subheader("Score a transaction")
    st.caption("Enter transaction details to get a live fraud risk score from the trained model.")

    c1, c2, c3 = st.columns(3)
    with c1:
        txn_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount", min_value=0.0, value=50000.0, step=1000.0)
        day = st.selectbox("Day of week", DAY_ORDER)
    with c2:
        old_orig = st.number_input("Sender balance before", min_value=0.0, value=50000.0, step=1000.0)
        new_orig = st.number_input("Sender balance after", min_value=0.0, value=0.0, step=1000.0)
        unusual_login = st.slider("Unusual login count", 0, 20, 10)
    with c3:
        old_dest = st.number_input("Receiver balance before", min_value=0.0, value=0.0, step=1000.0)
        new_dest = st.number_input("Receiver balance after", min_value=0.0, value=0.0, step=1000.0)
        is_merchant = st.checkbox("Receiver is a merchant account", value=False)

    if st.button("🔍 Check this transaction", type="primary"):
        if txn_type not in HIGH_RISK_TYPES:
            st.success(f"✅ **Low Risk** — {txn_type} transactions have a 0% historical fraud rate "
                       f"in this dataset (0 of {int(df[df['type']==txn_type].shape[0])} cases). "
                       f"The model wasn't trained on this type since there's no fraud signal to learn from it.")
        else:
            error_orig = new_orig - (old_orig - amount)
            error_dest = new_dest - (old_dest + amount)
            orig_wiped = int(old_orig > 0 and new_orig == 0)
            dest_unchanged = int(old_dest == 0 and new_dest == 0 and amount > 0)
            ratio = amount / (old_orig + 1)
            day_num = DAY_ORDER.index(day) + 1

            row = pd.DataFrame([{
                "amount": amount, "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
                "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
                "errorBalanceOrig": error_orig, "errorBalanceDest": error_dest,
                "origBalanceWiped": orig_wiped, "destBalanceUnchanged": dest_unchanged,
                "amountToBalanceRatio": ratio, "unusuallogin": unusual_login,
                "nameDestIsMerchant": int(is_merchant), "DayOfWeek": day_num,
                "type_TRANSFER": int(txn_type == "TRANSFER"),
            }])[FEATURES]

            if BEST_NAME == "Logistic Regression":
                prob = model.predict_proba(scaler.transform(row))[0, 1]
            else:
                prob = model.predict_proba(row)[0, 1]

            if prob >= 0.7:
                st.error(f"🚨 **High Risk** — fraud probability {prob*100:.1f}%")
            elif prob >= 0.3:
                st.warning(f"⚠️ **Medium Risk** — fraud probability {prob*100:.1f}%")
            else:
                st.success(f"✅ **Low Risk** — fraud probability {prob*100:.1f}%")

            flags = []
            if orig_wiped:
                flags.append("Sender's account was fully emptied by this transaction")
            if dest_unchanged:
                flags.append("Receiver balance shows no change despite a positive amount")
            if abs(error_orig) > 1:
                flags.append("Sender balance doesn't reconcile with the stated amount")
            if flags:
                st.markdown("**Flags raised:**")
                for f in flags:
                    st.markdown(f"- {f}")
