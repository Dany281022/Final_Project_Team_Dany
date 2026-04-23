# Weekly Sales Forecaster — Team Dany
## AIE1014 AI Applied Project | AIE1017 Generative AI & LLMs | Winter 2026

**Student:** Dany Deugoue | A00316024
**Role:**  MLOps Engineer
**Institution:** Cambrian College
**Date:** 19 April 2026

---

## Live Application

| Interface | URL |
|-----------|-----|
| Streamlit (Primary) | https://weekly-sales-forecaster-teamdany.streamlit.app/ |
| Render API + Dashboard | https://weekly-sales-forecaster-team-dany-aie1014.onrender.com |
| Swagger UI | https://weekly-sales-forecaster-team-dany-aie1014.onrender.com/docs |
| GitHub | https://github.com/Dany281022/Final_Project_Team_Dany|

---

## Model Performance

| Metric | v1 (Aggregated — rejected) | v2 (Per-Store — Production) |
|--------|----------------------------|------------------------------|
| R² Score | 0.2829 | **0.9812** |
| RMSE | $2,062,567 | **$73,473** |
| MAE | $1,488,586 | **$47,281** |
| CV Stability | — | R² = 0.9732 ± 0.0118 (TimeSeriesSplit, 5 folds) |
| Prediction range | — | $237,553 — $2,185,803 (1,935 real test rows) |

**Root cause of v1 failure:** aggregating 45 stores into 143 rows erased all store-level variance. Training per-store fixed this.

---

## Project Structure

```
Final_Project_Team_Dany/
│
├── app.py                      ← Streamlit app (4 tabs: Prediction, Dashboard, History, AI Advisor)
├── model.pkl                   ← RandomForestRegressor v2 — per-store (R²=0.9812, 8270 KB)
├── requirements.txt            ← Full dependencies
├── requirements_render.txt     ← Render-only dependencies
├── render.yaml                 ← Render deployment config
├── Dockerfile                  ← Docker container for Render
├── .env                        ← Environment variables (not committed — see .gitignore)
├── README.md                   ← This file
│
├── api/
│   ├── __init__.py
│   └── main.py                 ← FastAPI: /predict /explain /health /info /metrics
│
├── src/
│   ├── __init__.py
│   └── llm_client.py           ← OpenAI GPT-4o-mini + Ollama fallback (adapter pattern)
│
├── static/
│   └── index.html              ← HTML dashboard served by Render
│
├── code/
│   ├── data_pipeline.ipynb     ← Data preprocessing pipeline
│   ├── train_model.ipynb       ← Model v1 (aggregated — R²=0.2829, archived)
│   └── train_model_v2.ipynb    ← Model v2 (per-store — R²=0.9812, production)
│
├── data/
│   ├── raw/
│   │   └── Walmart.csv         ← Source dataset (Kaggle, 6,435 rows, 45 stores)
│   └── processed/
│       ├── X_train.csv         ← Training features (2,160 rows)
│       ├── X_test.csv          ← Test features (1,935 rows)
│       ├── y_train.csv         ← Training targets (log-transformed)
│       └── y_test.csv          ← Test targets (log-transformed)
│
├── models/
│   ├── rf_model_v2.pkl         ← RandomForest v2 reference copy
│   └── xgb_model.pkl           ← XGBoost (evaluated, not in production)
│
├── tests/
│   ├── test_predict.py         ← 14 model unit tests
│   └── test_integration.py     ← 10 pipeline integration tests
│
└── docs/
    ├── Milestone5_TeamDany_FinalReport.docx   ← Final technical report (10 sections)
    ├── UserGuide_TeamDany.docx                ← User guide + Quick Reference + FAQ
    └── Stakeholder_Documents_TeamDany.docx    ← Handoff letter + Feedback worksheet
```

---

## Prerequisites

- Python 3.10+
- pip
- Git

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Dany281022/Final_Project_Team_Dany.git
cd Final_Project_Team_Dany
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Edit `.env`:
```
OPENAI_API_KEY=sk-...
OLLAMA_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434
MODEL_PATH=model.pkl
```

---

## Running Locally

```bash
# Terminal 1 — Streamlit interface
streamlit run app.py

# Terminal 2 — FastAPI REST API
uvicorn api.main:app --reload
```

| Interface | Local URL |
|-----------|-----------|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |

---

## Running Tests

```bash
python -m pytest tests/ -v -s
```

**Expected result: 24/24 tests passing in ~23 seconds**

```
tests/test_integration.py  10 passed
tests/test_predict.py      14 passed
========================= 24 passed in 23.64s =========================
```

Key validations confirmed from real model.pkl:
- RMSE on 1,935 real test rows: **$73,473** (vs naive baseline $2,481,007)
- All 1,935 predictions positive, finite, unique
- Signal distribution: Stable 1,174 | Higher 372 | Lower 389
- % change range: -36.5% to +31.3%

---

## API Endpoints

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/` | GET | HTML dashboard (static/index.html) | HTML |
| `/health` | GET | Model + LLM status probe | `{status, model_loaded, r2_score, llm_ready}` |
| `/predict` | POST | Next-week sales forecast — 13 features | `{prediction, formatted, confidence, pct_change, signal, ms}` |
| `/explain` | POST | LLM business explanation of forecast | `{explanation, provider}` |
| `/info` | GET | Model metadata | `{model_type, features, r2, rmse, mae}` |
| `/metrics` | GET | Prometheus metrics | Prometheus text format |
| `/docs` | GET | Swagger UI | HTML |

### Example /predict request

```bash
curl -X POST https://weekly-sales-forecaster-team-dany-aie1014.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lag_1": 1500000, "lag_2": 1480000, "lag_4": 1450000,
    "lag_8": 1430000, "lag_12": 1400000, "lag_26": 1390000,
    "lag_52": 1380000, "ma_4": 1465000, "ma_12": 1430000,
    "std_4": 25000, "weekofyear": 15, "month": 4, "year": 2026
  }'
```

Expected response:
```json
{
  "prediction": 1421325.58,
  "formatted": "$1,421,325.58",
  "confidence": "$1,279,193 — $1,563,458",
  "pct_change": -5.24,
  "signal": "Lower demand",
  "response_time_ms": 51.34
}
```

---

## System Architecture

### Streamlit Cloud (Primary Interface)
```
User Browser → Streamlit Cloud → app.py + model.pkl + src/llm_client.py → Prediction + AI Advice
```

### Render (REST API + Docker)
```
User Browser → Render Docker → FastAPI api/main.py
                                    ↓
                   /predict  /explain  /health  /info  /metrics
                                    ↓
                          model.pkl + src/llm_client.py
```

### LLM Adapter Pattern
```
call_llm(prompt)
    ├── OpenAI GPT-4o-mini  (primary — if OPENAI_API_KEY set)
    │       ↓ fails
    └── Ollama llama3.2     (fallback — local)
            ↓ fails
        RuntimeError("All LLM providers failed")
```

---

## Model — v2 (Production)

| Parameter | Value |
|-----------|-------|
| Algorithm | RandomForestRegressor |
| Training strategy | Per-store — 45 stores × ~143 weeks = 6,435 rows |
| n_estimators | 200 (confirmed from model.pkl) |
| Target transform | log1p(y) at training / np.expm1() at inference |
| Train/Test split | Chronological — cutoff January 1, 2012 |
| Cross-validation | TimeSeriesSplit, 5 folds |

**13 Features:** `lag_1, lag_2, lag_4, lag_8, lag_12, lag_26, lag_52, ma_4, ma_12, std_4, weekofyear, month, year`

**Dominant feature:** `lag_52` (importance = 0.860) — year-over-year seasonality drives 86% of each forecast

---

## Deployment

### Streamlit Cloud
1. Force-add model.pkl: `git add --force model.pkl`
2. Connect repo at share.streamlit.io — set main file: `app.py`
3. Add secret: `OPENAI_API_KEY`

### Render (Docker)
1. Push code to GitHub — Dockerfile auto-detected
2. New Web Service → connect repo
3. Add env vars: `OPENAI_API_KEY`, `MODEL_PATH=model.pkl`
4. Verify: `GET /health` → `model_loaded: true`

---

## Known Issues

| Issue | Severity | Workaround |
|-------|----------|------------|
| Render free tier idles after 15 min — first request takes 30–60s | Low | Wait for page to load |
| Prediction history not persistent across browser sessions | Low | Download CSV from History tab before closing |
| One store at a time — no batch CSV upload | Low | Planned for future version |

---

## Documentation

All documentation is in the `docs/` folder:

| File | Contents |
|------|----------|
| `Final_Report_TeamDany.docx` | Full technical report — 10 sections, all metrics, architecture, tests, stakeholder engagement |
| `UserGuide_TeamDany.docx` | Step-by-step user guide with screenshots + Quick Reference Card + FAQ |
| `Stakeholder_Documents_TeamDany.docx` | Formal handoff letter + Feedback Worksheet (13 items, 5 fixed) |

---

## Stakeholder

- **Name:** Michael Thompson (AI-Simulated Retail Business Manager via GPT-4o-mini — approved by instructor)
- **Need:** Monday morning weekly sales forecasts for inventory and staffing decisions
- **Touchpoints:** 5 structured sessions across 15 weeks
- **UX fixes implemented:** 5 items fixed same day (April 8, 2026)
- **Value rating:** 4/5 — AI recommendations rated highest-value feature
---


*AIE1014 — AI Applied Project | AIE1017 — Generative AI and LLMs | Team Dany | Cambrian College | Winter 2026*
