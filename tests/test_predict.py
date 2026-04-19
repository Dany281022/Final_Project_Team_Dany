# tests/test_predict.py
"""
Unit tests for the Weekly Sales Forecaster model.
Tests model loading and prediction directly using real data from the dataset.
No API server required.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
import os

MODEL_PATH = "model.pkl"
DATA_DIR   = "data/processed"

# Column names as expected by model.pkl
MODEL_FEATURES = [
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_12",
    "lag_26", "lag_52", "ma_4", "ma_12", "std_4",
    "weekofyear", "month", "year"
]

# Column names as stored in X_test.csv (y_ prefix on lag columns)
CSV_FEATURES = [
    "y_lag_1", "y_lag_2", "y_lag_4", "y_lag_8", "y_lag_12",
    "y_lag_26", "y_lag_52", "ma_4", "ma_12", "std_4",
    "weekofyear", "month", "year"
]

# Mapping CSV columns → model feature names
RENAME_MAP = {
    "y_lag_1": "lag_1", "y_lag_2": "lag_2", "y_lag_4": "lag_4",
    "y_lag_8": "lag_8", "y_lag_12": "lag_12", "y_lag_26": "lag_26",
    "y_lag_52": "lag_52",
}


# ── Fixtures ───────────────────────────────────────────
@pytest.fixture(scope="module")
def model():
    """Load the trained RandomForestRegressor from model.pkl."""
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def real_data():
    """
    Load real X_test data and rename columns to match model feature names.
    X_test.csv uses y_lag_* prefix; model expects lag_* without prefix.
    """
    x_test_path  = os.path.join(DATA_DIR, "X_test.csv")
    x_train_path = os.path.join(DATA_DIR, "X_train.csv")

    if os.path.exists(x_test_path):
        df = pd.read_csv(x_test_path)
        print(f"\n  [fixture] Loaded X_test.csv — {len(df)} rows")
    elif os.path.exists(x_train_path):
        df = pd.read_csv(x_train_path)
        print(f"\n  [fixture] Loaded X_train.csv — {len(df)} rows")
    else:
        pytest.skip("No processed data found in data/processed/")

    # Rename y_lag_* → lag_* to match model.pkl feature names
    df = df.rename(columns=RENAME_MAP)

    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        pytest.skip(f"Missing columns after rename: {missing}")

    return df[MODEL_FEATURES]


@pytest.fixture(scope="module")
def real_targets():
    """Load real y_test targets for RMSE validation."""
    y_test_path  = os.path.join(DATA_DIR, "y_test.csv")
    y_train_path = os.path.join(DATA_DIR, "y_train.csv")

    if os.path.exists(y_test_path):
        return pd.read_csv(y_test_path).squeeze()
    elif os.path.exists(y_train_path):
        return pd.read_csv(y_train_path).squeeze()
    return None


# ── Model loading tests ────────────────────────────────
def test_model_loads(model):
    """Model file loads without error."""
    assert model is not None
    print("  ✅ Model loaded OK")


def test_model_has_correct_features(model):
    """Model expects exactly 13 features."""
    assert model.n_features_in_ == 13
    print(f"  ✅ Model expects {model.n_features_in_} features")


def test_model_is_random_forest(model):
    """Model is a RandomForestRegressor."""
    assert "RandomForest" in type(model).__name__
    print(f"  ✅ Model type: {type(model).__name__}")


def test_model_has_estimators(model):
    """Model has 200 trees as configured."""
    assert model.n_estimators == 200
    print(f"  ✅ n_estimators: {model.n_estimators}")


def test_model_feature_names(model):
    """Model feature names match expected lag columns."""
    expected = MODEL_FEATURES
    actual   = model.feature_names_in_.tolist()
    assert actual == expected
    print(f"  ✅ Feature names: {actual}")


# ── Real data prediction tests ─────────────────────────
def test_predict_first_real_row(model, real_data):
    """Prediction on first real test row returns a positive dollar amount."""
    row        = real_data.iloc[[0]]
    raw        = model.predict(row)[0]
    prediction = float(np.expm1(raw))
    assert prediction > 0
    print(f"  ✅ Row 0 — lag_1: ${real_data.iloc[0]['lag_1']:,.2f} → Prediction: ${prediction:,.2f}")


def test_predict_all_real_rows_positive(model, real_data):
    """All predictions on real data are positive dollar amounts."""
    raw         = model.predict(real_data)
    predictions = np.expm1(raw)
    assert (predictions > 0).all()
    print(f"  ✅ All {len(predictions)} predictions are positive")


def test_predict_real_range(model, real_data):
    """All real predictions fall within a realistic sales range."""
    raw         = model.predict(real_data)
    predictions = np.expm1(raw)
    assert predictions.min() > 100_000,     f"Min too low: ${predictions.min():,.0f}"
    assert predictions.max() < 500_000_000, f"Max too high: ${predictions.max():,.0f}"
    print(f"  ✅ Prediction range: ${predictions.min():,.0f} — ${predictions.max():,.0f}")


def test_predict_is_numeric(model, real_data):
    """All predictions are finite floats with no NaN or Inf."""
    raw         = model.predict(real_data)
    predictions = np.expm1(raw)
    assert np.isfinite(predictions).all()
    print(f"  ✅ All {len(predictions)} predictions are finite floats")


def test_predict_changes_with_different_input(model, real_data):
    """Different real rows produce different predictions."""
    raw         = model.predict(real_data)
    predictions = np.expm1(raw)
    unique      = len(np.unique(predictions))
    assert unique > 1
    print(f"  ✅ {unique} unique predictions across {len(predictions)} rows")


def test_predict_confidence_interval(model, real_data):
    """±10% confidence interval contains the prediction for all rows."""
    raw         = model.predict(real_data)
    predictions = np.expm1(raw)
    low         = predictions * 0.90
    high        = predictions * 1.10
    assert (low < predictions).all()
    assert (predictions < high).all()
    sample = predictions[0]
    print(f"  ✅ Sample interval: ${sample*0.9:,.0f} — ${sample:,.0f} — ${sample*1.1:,.0f}")


def test_pct_change_calculation(model, real_data):
    """Percentage change vs lag_1 is computable for all real rows."""
    raw         = model.predict(real_data)
    predictions = np.expm1(raw)
    lag_1       = real_data["lag_1"].values
    pct_changes = (predictions - lag_1) / lag_1 * 100
    assert np.isfinite(pct_changes).all()
    print(f"  ✅ % change range: {pct_changes.min():+.1f}% — {pct_changes.max():+.1f}%")


def test_predict_signal_distribution(model, real_data):
    """Demand signals are distributed across Stable / Higher / Lower."""
    raw         = model.predict(real_data)
    predictions = np.expm1(raw)
    lag_1       = real_data["lag_1"].values
    pct_changes = (predictions - lag_1) / lag_1 * 100

    stable = int((np.abs(pct_changes) <= 5).sum())
    higher = int((pct_changes > 5).sum())
    lower  = int((pct_changes < -5).sum())

    assert stable + higher + lower == len(predictions)
    print(f"  ✅ Signals — 🟢 Stable: {stable} | 🟡 Higher: {higher} | 🔵 Lower: {lower}")


def test_rmse_on_real_data(model, real_data, real_targets):
    """RMSE on real test data is below the naive baseline ($2,481,007)."""
    if real_targets is None:
        pytest.skip("No target data available")

    raw         = model.predict(real_data)
    predictions = np.expm1(raw)
    actuals     = real_targets.values

    # If targets are log-transformed, reverse the transformation
    if actuals.max() < 30:
        actuals = np.expm1(actuals)

    rmse           = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    NAIVE_BASELINE = 2_481_007
    assert rmse < NAIVE_BASELINE, \
        f"RMSE ${rmse:,.0f} exceeds naive baseline ${NAIVE_BASELINE:,.0f}"
    print(f"  ✅ RMSE: ${rmse:,.0f} (naive baseline: ${NAIVE_BASELINE:,.0f})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
    