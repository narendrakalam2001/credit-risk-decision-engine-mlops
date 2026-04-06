# ============================================================
# RISK ENGINE — Credit Risk ML System
# ============================================================
# Fintech-grade 3-tier decision system:
#   APPROVE  → low risk, auto-approve
#   REVIEW   → borderline, human analyst queue
#   DECLINE  → high risk, auto-decline
#
# Decision hierarchy:
#   1. Hard business rules (override ML)
#   2. ML model probability + risk bands
# ============================================================

import pandas as pd
import logging

from src.config import RISK_BANDS, MAX_INCOME_RATIO_RULE, MIN_INCOME_RULE, FAMILY_SIZE_RULE

logger = logging.getLogger(__name__)


# ============================================================
# RISK BAND — probability → LOW / MEDIUM / HIGH
# ============================================================

def get_risk_band(prob: float) -> str:
    for band, (low, high) in RISK_BANDS.items():
        if low <= prob < high:
            return band
    return "HIGH"


# ============================================================
# RISK ENGINE — row-level decisions
# ============================================================

def risk_engine(transaction_df: pd.DataFrame, probs, threshold: float) -> list:
    """
    For each applicant row → returns decision string.

    Rule priority:
      1. mortgage/income ratio > 10  → DECLINE (hard rule)
      2. income < 20 (000s)          → REVIEW  (flag for analyst)
      3. family >= 5                 → REVIEW  (flag for analyst)
      4. prob >= threshold           → DECLINE (ML model)
      5. prob >= threshold * 0.6     → REVIEW  (ML model borderline)
      6. else                        → APPROVE
    """
    decisions = []

    for idx, (_, row) in enumerate(transaction_df.iterrows()):

        p = probs[idx]

        # ── Hard rule: extreme mortgage burden ───────────────
        if "mortgage_income_ratio" in row and row["mortgage_income_ratio"] > MAX_INCOME_RATIO_RULE:
            decisions.append("DECLINE_RULE")
            continue

        # ── Soft rule: very low income ────────────────────────
        if "income" in row and row["income"] < MIN_INCOME_RULE:
            decisions.append("REVIEW_INCOME")
            continue

        # ── Soft rule: large family ───────────────────────────
        if "family" in row and row["family"] >= FAMILY_SIZE_RULE:
            decisions.append("REVIEW_FAMILY")
            continue

        # ── ML model decisions ────────────────────────────────
        if p >= threshold:
            decisions.append("DECLINE_MODEL")

        elif p >= threshold * 0.6:
            decisions.append("REVIEW_MODEL")

        else:
            decisions.append("APPROVE")

    return decisions


# ============================================================
# RISK SCORING — single applicant (for API)
# ============================================================

def score_applicant(row: dict, prob: float, threshold: float) -> dict:
    """
    Returns structured risk output for a single applicant.
    Used by FastAPI prediction endpoint.
    """
    risk_band = get_risk_band(prob)

    # ── Rule-based overrides ──────────────────────────────────
    rule_triggered = None

    mort_ratio = row.get("mortgage_income_ratio", 0)
    if mort_ratio > MAX_INCOME_RATIO_RULE:
        decision       = "DECLINE"
        rule_triggered = "HIGH_MORTGAGE_INCOME_RATIO"

    elif row.get("income", 999) < MIN_INCOME_RULE:
        decision       = "REVIEW"
        rule_triggered = "LOW_INCOME"

    elif row.get("family", 0) >= FAMILY_SIZE_RULE:
        decision       = "REVIEW"
        rule_triggered = "LARGE_FAMILY"

    # ── ML model decision ─────────────────────────────────────
    elif prob >= threshold:
        decision = "DECLINE"

    elif prob >= threshold * 0.6:
        decision = "REVIEW"

    else:
        decision = "APPROVE"

    return {
        "risk_probability": round(float(prob), 4),
        "risk_band":        risk_band,
        "decision":         decision,
        "rule_triggered":   rule_triggered,
    }