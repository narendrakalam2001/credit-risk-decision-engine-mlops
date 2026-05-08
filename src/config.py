# ============================================================
# CONFIGURATION — Credit Risk ML System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE   = 42
N_JOBS         = -1

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS             = 5
RANDOM_SEARCH_ITERS  = 20

# ── Feature selection ────────────────────────────────────────
SELECT_K = 10

# ── Outlier clipping ─────────────────────────────────────────
CLIP_FOLD = 1.5

# ── Feature engineering thresholds ───────────────────────────
ORDINAL_UNIQUE_THRESHOLD = 10

# ── Risk bands (probability → tier) ──────────────────────────
RISK_BANDS = {
    "LOW":    (0.00, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH":   (0.60, 1.01),
}

# ── Business rule thresholds ─────────────────────────────────
MAX_INCOME_RATIO_RULE  = 10.0
MIN_INCOME_RULE        = 20
FAMILY_SIZE_RULE       = 5

# ── PSI drift thresholds (NEW) ───────────────────────────────
PSI_MODERATE         = 0.10    # PSI >= 0.10 → moderate drift, monitor
PSI_HIGH             = 0.20    # PSI >= 0.20 → critical drift, retrain

# ── Score monitoring alert ───────────────────────────────────
SCORE_MEAN_ALERT     = 0.35

# ── Challenger promotion gates (NEW) ─────────────────────────
MIN_F1_IMPROVEMENT      = 0.005
MIN_ROCAUC_THRESHOLD    = 0.95
MAX_GENERALIZATION_GAP  = 0.10

# ── Paths ────────────────────────────────────────────────────
MODEL_DIR   = "risk_models"
METRICS_LOG = "risk_models/metrics_log.csv"
TESTS_DIR   = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)