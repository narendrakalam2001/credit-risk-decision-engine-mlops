# ============================================================
# DATA LOADER + FEATURE ENGINEERING — Credit Risk ML System
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Required columns for Bank Personal Loan dataset ──────────
REQUIRED_COLS = ["age", "income", "family", "ccavg", "mortgage", "personal_loan"]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:

    # ── Normalize column names ────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.lower()
    )

    # ── Drop irrelevant ID / zip columns ─────────────────────
    for col in ("id", "zip_code"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info("Dropped column: %s", col)

    # ── Check required columns ───────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Target validation ────────────────────────────────────
    if not set(df["personal_loan"].unique()).issubset({0, 1}):
        raise ValueError("Target 'personal_loan' must contain only 0 and 1")

    # ── Null check ───────────────────────────────────────────
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.warning("Dataset contains %d missing values — will be handled in preprocessing", nulls)

    # ── Minimum size check ───────────────────────────────────
    if df.shape[0] < 100:
        raise ValueError("Dataset too small for training (< 100 rows)")

    # ── Deduplication ────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)

    logger.info("Data validation passed  |  shape=%s  |  loan_rate=%.3f",
                df.shape, df["personal_loan"].mean())

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fintech-grade feature engineering for personal loan risk.
    All features map to real underwriting signals used by banks.
    """
    df = df.copy()

    # ── Debt & income ratios ──────────────────────────────────
    df["income_per_person"]       = df["income"] / df["family"].replace(0, 1)
    df["ccavg_income_ratio"]      = df["ccavg"]  / df["income"].replace(0, 1)
    df["mortgage_income_ratio"]   = df["mortgage"] / (df["income"] + 1)

    # ── Mortgage flag ─────────────────────────────────────────
    df["has_mortgage"] = (df["mortgage"] > 0).astype(int)

    # ── Age risk bins ─────────────────────────────────────────
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(float)

    df["near_retirement"] = (df["age"] >= 55).astype(int)

    # ── Digital engagement ────────────────────────────────────
    if "online" in df.columns and "creditcard" in df.columns:
        df["digital_usage"] = df["online"] + df["creditcard"]

    # ── Interaction features ──────────────────────────────────
    df["income_x_education"] = df["income"] * df.get("education", pd.Series(0, index=df.index))
    df["family_x_ccavg"]     = df["family"] * df["ccavg"]
    df["ccavg_x_mortgage"]   = df["ccavg"]  * df["has_mortgage"]

    # ── High-spend flag ───────────────────────────────────────
    ccavg_median = df["ccavg"].median()
    df["high_ccavg_flag"] = (df["ccavg"] > ccavg_median * 2).astype(int)

    # ── Drop leaky / redundant columns ───────────────────────
    if "experience" in df.columns:
        df.drop(columns=["experience"], inplace=True)

    logger.info("Feature engineering done  |  total columns=%d", df.shape[1])

    return df


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(df: pd.DataFrame, threshold: int = 10):
    """
    Auto-detect: ordinal / continuous / binary columns.
    Excludes target column 'personal_loan'.
    """
    ordinal_cols    = []
    continuous_cols = []
    binary_cols     = []

    for col in df.columns:
        if col == "personal_loan":
            continue

        dtype_name  = df[col].dtype.name
        n_unique    = df[col].nunique(dropna=False)

        if dtype_name in ("object", "category", "bool"):
            ordinal_cols.append(col)

        elif np.issubdtype(df[col].dtype, np.number):
            if n_unique == 2:
                binary_cols.append(col)
            elif 3 <= n_unique <= threshold:
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            ordinal_cols.append(col)

    logger.info("Feature types  |  ordinal=%d  continuous=%d  binary=%d",
                len(ordinal_cols), len(continuous_cols), len(binary_cols))

    return ordinal_cols, continuous_cols, binary_cols