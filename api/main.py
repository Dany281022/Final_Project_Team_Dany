# api/main.py
"""
FastAPI — Weekly Sales Forecaster
Serves dashboard + REST prediction + LLM explanation.
Deployed on Render.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title       = "Weekly Sales Forecaster API",
    description = "RandomForest + LLM insights for retail sales prediction",
    version     = "3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

Instrumentator().instrument(app).expose(app)

# ── Model ──────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
FEATURES   = [
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_12",
    "lag_26", "lag_52", "ma_4", "ma_12", "std_4",
    "weekofyear", "month", "year"
]

model = None

@app.on_event("startup")
def load_model() -> None:
    """Load trained model at startup."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[API] Model loaded — {model.n_estimators} trees")
    except FileNotFoundError:
        print(f"[API] WARNING: {MODEL_PATH} not found")


# ── Schemas ────────────────────────────────────────────
class PredictRequest(BaseModel):
    lag_1:      float
    lag_2:      float
    lag_4:      float
    lag_8:      float
    lag_12:     float
    lag_26:     float
    lag_52:     float
    ma_4:       float
    ma_12:      float
    std_4:      float
    weekofyear: int
    month:      int
    year:       int

class PredictResponse(BaseModel):
    prediction:       float
    formatted:        str
    confidence:       str
    pct_change:       float
    signal:           str
    response_time_ms: float

class ExplainRequest(BaseModel):
    prediction:  float
    pct_change:  float
    signal:      str
    lag_1:       float
    weekofyear:  int
    month:       int

class ExplainResponse(BaseModel):
    explanation: str
    provider:    str

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    n_estimators: int
    r2_score:     float
    llm_ready:    bool


# ── Endpoints ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Health probe for Render."""
    return HealthResponse(
        status       = "ok",
        model_loaded = model is not None,
        n_estimators = model.n_estimators if model else 0,
        r2_score     = 0.9812,
        llm_ready    = bool(os.getenv("OPENAI_API_KEY", "")),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict next-week retail sales from 13 lag features.
    Model trained on log1p(y) — response in dollar scale via expm1.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        start      = time.time()
        X          = pd.DataFrame([request.model_dump()])[FEATURES]
        raw        = model.predict(X)[0]
        prediction = float(np.expm1(raw))
        elapsed_ms = round((time.time() - start) * 1000, 2)

        lag_1      = request.lag_1
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

        return PredictResponse(
            prediction       = prediction,
            formatted        = f"${prediction:,.2f}",
            confidence       = confidence,
            pct_change       = round(pct_change, 2),
            signal           = signal,
            response_time_ms = elapsed_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse, tags=["LLM"])
def explain(request: ExplainRequest) -> ExplainResponse:
    """
    Generate LLM-powered business explanation for the sales forecast.
    Uses OpenAI with automatic Ollama fallback.
    """
    try:
        from src.llm_client import call_llm, build_sales_prompt
        prompt      = build_sales_prompt(
            prediction = request.prediction,
            pct_change = request.pct_change,
            signal     = request.signal,
            lag_1      = request.lag_1,
            weekofyear = request.weekofyear,
            month      = request.month,
        )
        explanation = call_llm(prompt)
        provider    = "OpenAI" if os.getenv("OPENAI_API_KEY") else "Ollama"
        return ExplainResponse(explanation=explanation, provider=provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", tags=["Monitoring"])
def info() -> dict:
    """Return model metadata."""
    return {
        "model_type":  "RandomForestRegressor",
        "num_features": 13,
        "features":    FEATURES,
        "performance": {"r2": 0.9812, "rmse": 73473, "mae": 47281},
        "version":     "3.0.0",
        "llm_support": True,
    }


# ── Dashboard ──────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve the main dashboard HTML page."""
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)