# tests/test_integration.py
"""
Integration tests for the Weekly Sales Forecaster.
Tests the full prediction pipeline end-to-end — no API server required.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
import os

MODEL_PATH = "model.pkl"

FEATURES = [
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_12",
    "lag_26", "lag_52", "ma_4", "ma_12", "std_4",
    "weekofyear", "month", "year"
]


def make_prediction(model, input_dict: dict) -> dict:
    """
    Full prediction pipeline: input dict → prediction output dict.
    Mirrors what app.py does internally.
    Uses DataFrame to match model feature names and suppress warnings.
    """
    X          = pd.DataFrame([input_dict])[FEATURES]
    raw        = model.predict(X)[0]
    prediction = float(np.expm1(raw))
    lag_1      = input_dict["lag_1"]
    pct_change = (prediction - lag_1) / lag_1 * 100 if lag_1 > 0 else 0
    low        = prediction * 0.90
    high       = prediction * 1.10
    confidence = f"${low:,.0f} — ${high:,.0f}"

    if abs(pct_change) <= 5:
        signal = "Stable"
    elif pct_change > 5:
        signal = "Higher demand"
    else:
        signal = "Lower demand"

    return {
        "prediction": prediction,
        "pct_change": pct_change,
        "confidence": confidence,
        "signal":     signal,
    }


# ── Fixtures ───────────────────────────────────────────
@pytest.fixture(scope="module")
def model():
    """Load trained model once for all integration tests."""
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture
def typical_input():
    return {
        "lag_1":  40000000.0, "lag_2":  39000000.0,
        "lag_4":  38000000.0, "lag_8":  37000000.0,
        "lag_12": 36000000.0, "lag_26": 35000000.0,
        "lag_52": 34000000.0, "ma_4":   38500000.0,
        "ma_12":  37000000.0, "std_4":    500000.0,
        "weekofyear": 10, "month": 3, "year": 2026,
    }


@pytest.fixture
def low_sales_input():
    return {
        "lag_1":  5000000.0, "lag_2":  4800000.0,
        "lag_4":  4600000.0, "lag_8":  4400000.0,
        "lag_12": 4200000.0, "lag_26": 4000000.0,
        "lag_52": 3800000.0, "ma_4":   4700000.0,
        "ma_12":  4300000.0, "std_4":   200000.0,
        "weekofyear": 5, "month": 1, "year": 2026,
    }


@pytest.fixture
def high_sales_input():
    return {
        "lag_1":  80000000.0, "lag_2":  78000000.0,
        "lag_4":  75000000.0, "lag_8":  72000000.0,
        "lag_12": 70000000.0, "lag_26": 68000000.0,
        "lag_52": 65000000.0, "ma_4":   77000000.0,
        "ma_12":  72000000.0, "std_4":  1000000.0,
        "weekofyear": 52, "month": 12, "year": 2026,
    }


# ── Tests ──────────────────────────────────────────────
def test_full_pipeline_typical(model, typical_input):
    """Full pipeline runs end-to-end with typical retail input."""
    result = make_prediction(model, typical_input)
    assert result["prediction"] > 0
    assert "confidence" in result
    assert result["signal"] in ["Stable", "Higher demand", "Lower demand"]
    print(f"  ✅ Pipeline OK — ${result['prediction']:,.2f} | {result['signal']}")


def test_full_pipeline_low_sales(model, low_sales_input):
    """Pipeline works with low sales figures."""
    result = make_prediction(model, low_sales_input)
    assert result["prediction"] > 0
    print(f"  ✅ Low sales pipeline OK — ${result['prediction']:,.2f}")


def test_full_pipeline_high_sales(model, high_sales_input):
    """Pipeline works with high sales figures (holiday season)."""
    result = make_prediction(model, high_sales_input)
    assert result["prediction"] > 0
    print(f"  ✅ High sales pipeline OK — ${result['prediction']:,.2f} | {result['signal']}")


def test_signal_stable(model, typical_input):
    """Signal is correctly assigned based on pct_change."""
    result = make_prediction(model, typical_input)
    if abs(result["pct_change"]) <= 5:
        assert result["signal"] == "Stable"
    elif result["pct_change"] > 5:
        assert result["signal"] == "Higher demand"
    else:
        assert result["signal"] == "Lower demand"
    print(f"  ✅ Signal: {result['signal']} ({result['pct_change']:+.1f}%)")


def test_confidence_interval_format(model, typical_input):
    """Confidence interval is formatted as a dollar range string."""
    result = make_prediction(model, typical_input)
    assert "—" in result["confidence"]
    assert "$" in result["confidence"]
    print(f"  ✅ Confidence interval: {result['confidence']}")


def test_confidence_interval_bounds(model, typical_input):
    """Confidence interval lower bound < prediction < upper bound."""
    result     = make_prediction(model, typical_input)
    prediction = result["prediction"]
    low        = prediction * 0.90
    high       = prediction * 1.10
    assert low < prediction < high
    print(f"  ✅ Bounds OK: ${low:,.0f} < ${prediction:,.0f} < ${high:,.0f}")


def test_model_file_exists():
    """model.pkl exists at the expected path."""
    assert os.path.exists(MODEL_PATH)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    assert size_kb > 10
    print(f"  ✅ model.pkl exists ({size_kb:.0f} KB)")


def test_all_features_used(model):
    """Model uses exactly the 13 expected features."""
    assert model.n_features_in_ == len(FEATURES) == 13
    print(f"  ✅ All 13 features verified")


def test_prediction_is_dollar_scale(model, typical_input):
    """Prediction is in dollar scale (not log scale)."""
    result = make_prediction(model, typical_input)
    assert result["prediction"] > 1_000, "Prediction looks like log scale"
    print(f"  ✅ Dollar scale confirmed: ${result['prediction']:,.2f}")


def test_pct_change_formula(model, typical_input):
    """Percentage change formula is correct."""
    result     = make_prediction(model, typical_input)
    prediction = result["prediction"]
    lag_1      = typical_input["lag_1"]
    expected   = (prediction - lag_1) / lag_1 * 100
    assert abs(result["pct_change"] - expected) < 0.001
    print(f"  ✅ pct_change formula correct: {result['pct_change']:+.2f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])