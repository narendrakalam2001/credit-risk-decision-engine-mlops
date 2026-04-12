# ============================================================
# MONITORING DASHBOARD — Credit Risk ML System
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("💳 Credit Risk Monitoring Dashboard")

API_URL = os.getenv(
    "CREDIT_RISK_API_URL",
    "https://credit-risk-decision-engine-mlops.onrender.com"
) + "/predic

# ── Absolute paths — works both locally and on Streamlit Cloud ──
# Streamlit Cloud runs from project root, but use __file__ to be safe
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── PSI alert thresholds ──────────────────────────────────────
PSI_MODERATE = 0.10
PSI_HIGH     = 0.20

# ── Score drift alert threshold ───────────────────────────────
SCORE_MEAN_ALERT = 0.35


# ============================================================
# SIDEBAR — LIVE PREDICTION
# ============================================================

st.sidebar.header("🔮 Predict Applicant Risk")

age        = st.sidebar.number_input("Age",        value=35,    min_value=18, max_value=80)
income     = st.sidebar.number_input("Income (K)", value=60.0,  min_value=0.0)
family     = st.sidebar.number_input("Family Size",value=2,     min_value=1,  max_value=10, step=1)
ccavg      = st.sidebar.number_input("CC Avg (K)", value=2.0,   min_value=0.0)
education  = st.sidebar.selectbox("Education", [1, 2, 3],
                                   format_func=lambda x: {1:"Undergrad",2:"Graduate",3:"Advanced"}[x])
mortgage   = st.sidebar.number_input("Mortgage (K)", value=0.0, min_value=0.0)
online     = st.sidebar.selectbox("Online Banking", [0, 1])
creditcard = st.sidebar.selectbox("Has Credit Card", [0, 1])

if st.sidebar.button("Predict Risk"):
    payload = {
        "age": age, "income": income, "family": family,
        "ccavg": ccavg, "education": education, "mortgage": mortgage,
        "online": online, "creditcard": creditcard,
        "securities_account": 0, "cd_account": 0
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result   = response.json()
            decision = result["decision"]
            color    = {"APPROVE": "green", "REVIEW": "orange", "DECLINE": "red"}.get(decision, "gray")
            st.sidebar.markdown(f"**Risk Probability:** `{result['risk_probability']}`")
            st.sidebar.markdown(f"**Risk Band:** `{result['risk_band']}`")
            st.sidebar.markdown(
                f"<h3 style='color:{color}'>Decision: {decision}</h3>",
                unsafe_allow_html=True
            )
            if result.get("rule_triggered"):
                st.sidebar.warning(f"Rule triggered: {result['rule_triggered']}")
        else:
            st.sidebar.error(f"API error: {response.status_code}")
            st.sidebar.code(response.text[:300])
    except Exception as e:
        st.sidebar.error(f"Connection error: {e}")
        st.sidebar.caption(f"API URL: {API_URL}")


# ============================================================
# PATHS
# ============================================================

MONITOR_PATH    = os.path.join(BASE_DIR, "risk_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "risk_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "risk_models", "challenger_log.json")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(MONITOR_PATH):
    df_monitor = pd.read_csv(MONITOR_PATH)

    # ── Alert 1: High average risk score ─────────────────────
    if "score" in df_monitor.columns:
        avg_score = df_monitor["score"].mean()
        if avg_score > SCORE_MEAN_ALERT:
            st.error(
                f"🔴 HIGH RISK SCORE ALERT: Average risk probability = {avg_score:.4f} "
                f"(threshold: {SCORE_MEAN_ALERT}). "
                f"Review applicant pool — unusual risk concentration detected."
            )
            alerts_found = True

    # ── Alert 2: High decline rate ────────────────────────────
    if "decision" in df_monitor.columns:
        decline_rate = df_monitor["decision"].str.contains("DECLINE").mean()
        if decline_rate > 0.25:
            st.warning(
                f"🟡 HIGH DECLINE RATE: {decline_rate:.1%} of applicants declined "
                f"(expected < 25%). Possible data shift or model drift."
            )
            alerts_found = True

        # ── Alert 3: High review queue ────────────────────────
        review_rate = df_monitor["decision"].str.contains("REVIEW").mean()
        if review_rate > 0.20:
            st.warning(
                f"🟡 HIGH REVIEW QUEUE: {review_rate:.1%} applicants in REVIEW "
                f"(expected < 20%). Analyst capacity may be exceeded."
            )
            alerts_found = True

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi.columns:
        max_psi     = df_psi["drift_score"].max()
        top_feature = df_psi.iloc[0]["feature"] if "feature" in df_psi.columns else "unknown"

        if max_psi >= PSI_HIGH:
            st.error(
                f"🔴 CRITICAL FEATURE DRIFT: PSI = {max_psi:.4f} on '{top_feature}' "
                f"(>= {PSI_HIGH}). Model retraining recommended immediately."
            )
            alerts_found = True
        elif max_psi >= PSI_MODERATE:
            st.warning(
                f"🟡 MODERATE FEATURE DRIFT: PSI = {max_psi:.4f} on '{top_feature}' "
                f"({PSI_MODERATE}–{PSI_HIGH}). Monitor closely."
            )
            alerts_found = True

if not alerts_found:
    st.success("✅ All systems normal — no alerts triggered")


# ============================================================
# SECTION 2 — CHALLENGER MODEL STATUS
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        challenger_log = json.load(f)

    if challenger_log:
        # ── Latest result ─────────────────────────────────────
        latest = challenger_log[-1]
        decision_color = "green" if latest["decision"] == "PROMOTED" else "red"
        icon           = "✅" if latest["decision"] == "PROMOTED" else "❌"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Challenger",  latest.get("challenger_name", "—"))
        with col2:
            st.metric("Challenger F1",      latest.get("challenger_f1",  "—"))
        with col3:
            st.metric("Champion F1",        latest.get("champion_f1",    "—") or "First Run")

        st.markdown(
            f"<h4 style='color:{decision_color}'>"
            f"{icon} Decision: {latest['decision']} — {latest.get('reason','')}"
            f"</h4>",
            unsafe_allow_html=True
        )

        # ── Full history table ────────────────────────────────
        if len(challenger_log) > 1:
            with st.expander("View full challenger history"):
                history_df = pd.DataFrame(challenger_log)
                display_cols = [c for c in [
                    "evaluated_at","challenger_name","challenger_f1",
                    "champion_name","champion_f1","decision","reason"
                ] if c in history_df.columns]
                st.dataframe(history_df[display_cols])
else:
    st.info("No challenger log found. Run training pipeline to generate.")


# ============================================================
# SECTION 3 — KPI METRICS
# ============================================================

st.markdown("---")
st.subheader("📊 Model Performance KPIs")

if os.path.exists(MONITOR_PATH):
    df = pd.read_csv(MONITOR_PATH)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if "label" in df.columns:
            st.metric("Loan Approval Rate", f"{df['label'].mean():.1%}")
    with col2:
        if "decision" in df.columns:
            st.metric("Auto-Approve Rate", f"{(df['decision']=='APPROVE').mean():.1%}")
    with col3:
        if "decision" in df.columns:
            st.metric("Review Queue Rate", f"{df['decision'].str.contains('REVIEW').mean():.1%}")
    with col4:
        if "decision" in df.columns:
            st.metric("Decline Rate", f"{df['decision'].str.contains('DECLINE').mean():.1%}")

    # ── Charts ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Score Distribution")
        if "score" in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df["score"], bins=50, alpha=0.7, color="steelblue")
            ax.axvline(0.30, color="orange", linestyle="--", label="LOW/MED boundary")
            ax.axvline(0.60, color="red",    linestyle="--", label="MED/HIGH boundary")
            if df["score"].mean() > SCORE_MEAN_ALERT:
                ax.axvline(df["score"].mean(), color="darkred", linestyle="-",
                           linewidth=2, label=f"⚠ Mean={df['score'].mean():.3f}")
            ax.legend(fontsize=8)
            st.pyplot(fig)

    with col_b:
        st.subheader("Decision Distribution")
        if "decision" in df.columns:
            st.bar_chart(df["decision"].value_counts())

    if "risk_band" in df.columns:
        st.subheader("Risk Band Breakdown")
        st.bar_chart(df["risk_band"].value_counts())

    st.subheader("Score Statistics")
    if "score" in df.columns:
        st.write(df["score"].describe().to_frame().T.round(4))

else:
    st.warning("No monitor scores found. Run training pipeline first.")


# ============================================================
# SECTION 4 — PSI DRIFT REPORT
# ============================================================

st.markdown("---")
st.subheader("📉 Feature Drift Report (PSI)")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi.columns:
        def _psi_flag(val):
            if val >= PSI_HIGH:
                return "🔴 CRITICAL"
            elif val >= PSI_MODERATE:
                return "🟡 MODERATE"
            return "🟢 OK"

        df_psi["status"] = df_psi["drift_score"].apply(_psi_flag)
        st.dataframe(df_psi.head(10), use_container_width=True)
else:
    st.info("No drift report found.")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

if os.path.exists(LOG_PATH):
    st.markdown("---")
    st.subheader("📋 Recent Predictions")
    log_df = pd.read_csv(LOG_PATH)
    st.dataframe(log_df.tail(20))
else:
    st.info("No prediction logs yet. Send requests to the API.")