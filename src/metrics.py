# ============================================================
# METRICS — Credit Risk ML System
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(
    y_true:           np.ndarray,
    y_prob:           np.ndarray,
    target_recall:    float = None,
    target_precision: float = None
) -> float:
    """
    Default: maximize F1.
    Optional: constrain on recall or precision first.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx  = np.nanargmax(f1_scores)
    best_thr  = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    if target_recall is not None:
        idxs = np.where(recall >= target_recall)[0]
        if idxs.size > 0:
            chosen = idxs[np.argmax(precision[idxs])]
            best_thr = thresholds[chosen] if chosen < len(thresholds) else 0.5

    if target_precision is not None:
        idxs = np.where(precision >= target_precision)[0]
        if idxs.size > 0:
            chosen = idxs[np.argmax(recall[idxs])]
            best_thr = thresholds[chosen] if chosen < len(thresholds) else 0.5

    logger.info("Best threshold: %.4f  |  precision=%.4f  recall=%.4f  f1=%.4f",
                best_thr,
                precision[best_idx],
                recall[best_idx],
                f1_scores[best_idx])

    return float(best_thr)


# ============================================================
# PSI — Population Stability Index
# ============================================================

def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.1   → no significant shift (stable)
    PSI 0.1–0.2 → moderate shift (monitor closely)
    PSI > 0.2   → major shift (retrain recommended)

    Correct approach:
      1. Compute quantile bin EDGES from `expected` (reference)
      2. Bin BOTH `expected` and `actual` using those SAME edges
      3. Compare bin proportions

    Common bug to avoid:
      Ranking both distributions independently → both become
      uniform → PSI always ~0. This is wrong.
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        # ── Step 1: compute bin edges from reference distribution ──
        quantiles  = np.linspace(0, 100, buckets + 1)
        bin_edges  = np.percentile(expected, quantiles)

        # Make edges unique to avoid empty bins — add small epsilon
        bin_edges  = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        # Extend edges to cover full range of actual distribution
        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        # ── Step 2: bin BOTH using the SAME edges ─────────────────
        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        # ── Step 3: convert to proportions, avoid div-by-zero ─────
        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        # ── Step 4: PSI formula ────────────────────────────────────
        psi_value = float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

        return psi_value

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# KS STATISTIC
# ============================================================

def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ============================================================
# RECALL @ K
# ============================================================

def recall_at_k(y_true, y_prob, k: float = 0.05) -> float:
    df = (
        pd.DataFrame({"y": y_true, "p": y_prob})
        .sort_values("p", ascending=False)
    )
    top_n = int(len(df) * k)
    return float(df.iloc[:top_n]["y"].sum() / (df["y"].sum() + 1e-9))


# ============================================================
# LIFT @ K
# ============================================================

def lift_at_k(y_true, y_prob, k: float = 0.05) -> float:
    base = np.mean(y_true)
    return float(recall_at_k(y_true, y_prob, k) / (base + 1e-9))


# ============================================================
# COST-SENSITIVE EVALUATION  (fintech-grade)
# ============================================================

def cost_sensitive_evaluation(
    X_test,
    y_true,
    y_pred,
    loan_amount_col: str   = "income",
    loss_given_default: float = 0.6,    # 60 % LGD — industry standard
    review_cost_per_case: float = 50.0  # analyst review cost (₹)
) -> dict:
    """
    Expected Credit Loss (ECL) approach:

        ECL = PD × LGD × EAD

    where:
        PD  = predicted probability (bad loan)
        LGD = loss given default (fixed assumption)
        EAD = exposure at default (income proxy here)

    Also computes:
        false_negative_loss — missed bad loans
        false_positive_cost — unnecessary review of good loans
    """

    df = X_test.copy()
    df["y_true"] = y_true.values if hasattr(y_true, "values") else y_true
    df["y_pred"] = y_pred

    # ── Missed bad loans (FN) → actual loss ──────────────────
    fn_mask     = (df["y_true"] == 1) & (df["y_pred"] == 0)
    fn_exposure = df.loc[fn_mask, loan_amount_col].sum() if loan_amount_col in df.columns else fn_mask.sum()
    fn_loss     = float(fn_exposure * loss_given_default)

    # ── False positive review cost ────────────────────────────
    fp_mask  = (df["y_true"] == 0) & (df["y_pred"] == 1)
    fp_cost  = float(fp_mask.sum() * review_cost_per_case)

    total_loss = fn_loss + fp_cost

    result = {
        "false_negative_count": int(fn_mask.sum()),
        "false_positive_count": int(fp_mask.sum()),
        "estimated_fn_loss":    round(fn_loss, 2),
        "review_cost":          round(fp_cost, 2),
        "total_estimated_loss": round(total_loss, 2),
    }

    logger.info(
        "Cost eval  |  FN=%d  FP=%d  FN_loss=%.2f  review_cost=%.2f  total=%.2f",
        result["false_negative_count"],
        result["false_positive_count"],
        result["estimated_fn_loss"],
        result["review_cost"],
        result["total_estimated_loss"]
    )

    return result


# ============================================================
# DRIFT REPORT — feature mean shift
# ============================================================

def simple_drift_report(X_ref: "pd.DataFrame", X_new: "pd.DataFrame", top_n: int = 10):
    diffs = (X_ref.mean() - X_new.mean()).abs()
    rel   = (diffs / (X_ref.std().replace(0, 1))).sort_values(ascending=False)
    return rel.head(top_n)