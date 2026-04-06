# ============================================================
# PYTEST UNIT TESTS — Credit Risk ML System
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#
# These tests are NOT imported anywhere — pytest discovers
# and runs them automatically via:  pytest tests/
#
# They test individual modules in isolation:
#   - Clipper transformer
#   - build_preprocessors
#   - detect_feature_types
#   - detect_leakage
#   - tune_threshold
#   - risk_engine / score_applicant
#   - psi
# ============================================================

import sys
import os

# ── Make sure project root is on path ────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd

from src.preprocessing  import Clipper, build_preprocessors
from src.data_loader    import detect_feature_types, add_engineered_features
from src.leakage_check  import detect_leakage
from src.metrics        import tune_threshold, psi
from src.risk_engine    import get_risk_band, score_applicant
from src.config         import RISK_BANDS


# ============================================================
# CLIPPER TESTS
# ============================================================

class TestClipper:

    def test_fit_transform_shape(self):
        """Output shape must match input shape."""
        X = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        Xt = clip.transform(X)
        assert Xt.shape == X.shape

    def test_clips_outliers(self):
        """Extreme values must be clipped."""
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        Xt = clip.transform(X)
        assert Xt.max() < 9999.0, "Outlier should have been clipped"

    def test_no_change_on_normal_data(self):
        """Values within IQR range must not be changed."""
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        Xt = clip.transform(X)
        np.testing.assert_array_almost_equal(X, Xt, decimal=3)

    def test_1d_input(self):
        """1D array should be handled without error."""
        X = np.array([1.0, 2.0, 3.0, 1000.0])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        Xt = clip.transform(X)
        assert Xt.shape[0] == 4


# ============================================================
# PREPROCESSOR TESTS
# ============================================================

class TestBuildPreprocessors:

    def _sample_df(self):
        return pd.DataFrame({
            "age":      [25, 35, 45, 55, 65],
            "income":   [30.0, 60.0, 90.0, 120.0, 150.0],
            "family":   [1, 2, 3, 4, 5],
            "education":[1, 2, 3, 1, 2],
            "online":   [0, 1, 0, 1, 0],
        })

    def test_returns_four_outputs(self):
        """build_preprocessors must return (pre_scaled, pre_unscaled, cat_idx, feat_order)."""
        df = self._sample_df()
        ord_cols  = ["education", "family"]
        cont_cols = ["age", "income"]
        bin_cols  = ["online"]
        result = build_preprocessors(ord_cols, cont_cols, bin_cols, df)
        assert len(result) == 4

    def test_categorical_indices_are_list(self):
        df = self._sample_df()
        ord_cols  = ["education"]
        cont_cols = ["age", "income"]
        bin_cols  = ["online"]
        _, _, cat_idx, _ = build_preprocessors(ord_cols, cont_cols, bin_cols, df)
        assert isinstance(cat_idx, list)

    def test_feature_order_coverage(self):
        """feature_order must contain all input columns."""
        df = self._sample_df()
        ord_cols  = ["education", "family"]
        cont_cols = ["age", "income"]
        bin_cols  = ["online"]
        _, _, _, feat_order = build_preprocessors(ord_cols, cont_cols, bin_cols, df)
        for col in ord_cols + cont_cols + bin_cols:
            assert col in feat_order


# ============================================================
# FEATURE TYPE DETECTION TESTS
# ============================================================

class TestDetectFeatureTypes:

    def test_binary_detected(self):
        df = pd.DataFrame({
            "flag":          [0, 1, 0, 1, 0],
            "personal_loan": [0, 1, 0, 0, 1]
        })
        _, _, bin_cols = detect_feature_types(df, threshold=10)
        assert "flag" in bin_cols

    def test_target_excluded(self):
        df = pd.DataFrame({
            "income":        [10.0, 20.0, 30.0, 40.0, 50.0],
            "personal_loan": [0, 1, 0, 0, 1]
        })
        ord_cols, cont_cols, bin_cols = detect_feature_types(df, threshold=10)
        all_cols = ord_cols + cont_cols + bin_cols
        assert "personal_loan" not in all_cols

    def test_continuous_detected(self):
        df = pd.DataFrame({
            "income":        [10.5, 20.1, 55.3, 80.0, 110.5, 200.3,
                              15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
            "personal_loan": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        _, cont_cols, _ = detect_feature_types(df, threshold=10)
        assert "income" in cont_cols


# ============================================================
# LEAKAGE CHECK TESTS
# ============================================================

class TestDetectLeakage:

    def test_catches_identical_feature(self):
        """Feature identical to target should be flagged."""
        X = pd.DataFrame({"a": [1, 0, 1, 0, 1]})
        y = pd.Series(        [1, 0, 1, 0, 1])
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) > 0
        assert any("a" in w for w in warnings)

    def test_no_false_positives_on_clean_data(self):
        """Normal features must not be flagged."""
        np.random.seed(42)
        X = pd.DataFrame({"income": np.random.uniform(20, 200, 100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) == 0

    def test_catches_high_correlation(self):
        """Near-perfect correlation (>= threshold) must be flagged."""
        # Use threshold 0.85 — arange(50) vs binary split gives ~0.87 corr
        vals = np.arange(50, dtype=float)
        X    = pd.DataFrame({"leaky_feat": vals})
        y    = pd.Series((vals > 25).astype(int))
        warnings = detect_leakage(X, y, threshold_corr=0.85)
        assert len(warnings) > 0


# ============================================================
# THRESHOLD TUNING TESTS
# ============================================================

class TestTuneThreshold:

    def test_returns_float_in_range(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200)
        thr = tune_threshold(y_true, y_prob)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_perfect_prob_gives_low_threshold(self):
        """Perfect predictions → threshold should be <= 0.9 (not forced above it)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        thr = tune_threshold(y_true, y_prob)
        # With perfect separation, best F1 threshold is at or below 0.9
        assert thr <= 0.9, f"Threshold {thr} should be <= 0.9 for perfectly separated data"


# ============================================================
# PSI TESTS
# ============================================================

class TestPSI:

    def test_identical_distributions(self):
        """PSI of identical distributions should be near 0."""
        x = np.random.normal(0, 1, 500)
        score = psi(x, x)
        assert score < 0.05, f"PSI for identical distributions should be ~0, got {score}"

    def test_shifted_distribution_higher_psi(self):
        """Shifted distribution should have higher PSI than identical."""
        # Fixed seed + different sizes avoids rank-collapse to 0.0
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 1000)
        new = rng.normal(3, 1, 1000)   # mean shift of 3 sigma — clearly different
        score_same    = psi(ref, ref)
        score_shifted = psi(ref, new)
        assert score_shifted > score_same, (
            f"Shifted PSI ({score_shifted:.4f}) should be > same PSI ({score_same:.4f})"
        )


# ============================================================
# RISK ENGINE TESTS
# ============================================================

class TestRiskEngine:

    def test_get_risk_band_low(self):
        assert get_risk_band(0.10) == "LOW"

    def test_get_risk_band_medium(self):
        assert get_risk_band(0.45) == "MEDIUM"

    def test_get_risk_band_high(self):
        assert get_risk_band(0.75) == "HIGH"

    def test_score_applicant_approve(self):
        row    = {"income": 100.0, "family": 2, "mortgage_income_ratio": 0.5}
        result = score_applicant(row, prob=0.05, threshold=0.5)
        assert result["decision"] == "APPROVE"
        assert result["risk_band"] == "LOW"

    def test_score_applicant_decline_rule(self):
        """Extreme mortgage/income ratio must trigger DECLINE rule."""
        row    = {"income": 20.0, "family": 2, "mortgage_income_ratio": 15.0}
        result = score_applicant(row, prob=0.05, threshold=0.5)
        assert result["decision"] == "DECLINE"
        assert result["rule_triggered"] == "HIGH_MORTGAGE_INCOME_RATIO"

    def test_score_applicant_review_model(self):
        """Borderline probability must trigger REVIEW."""
        row    = {"income": 80.0, "family": 2, "mortgage_income_ratio": 1.0}
        result = score_applicant(row, prob=0.35, threshold=0.5)
        assert result["decision"] == "REVIEW"

    def test_score_applicant_decline_model(self):
        """High probability must trigger DECLINE via model."""
        row    = {"income": 80.0, "family": 2, "mortgage_income_ratio": 1.0}
        result = score_applicant(row, prob=0.9, threshold=0.5)
        assert result["decision"] == "DECLINE"
        assert result["rule_triggered"] is None

    def test_output_keys_complete(self):
        """Response must always contain all 4 keys."""
        row    = {"income": 60.0, "family": 2, "mortgage_income_ratio": 1.0}
        result = score_applicant(row, prob=0.2, threshold=0.5)
        for key in ("risk_probability", "risk_band", "decision", "rule_triggered"):
            assert key in result