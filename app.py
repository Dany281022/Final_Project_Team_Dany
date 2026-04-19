"""
Weekly Sales Forecaster — Team Dany
AIE1014 - AI Applied Project Course | Milestone 5

Deployed ML Prediction Application — Combined Streamlit + Model + LLM
Stakeholder: Retail Business Manager
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_TITLE  = "📈 Weekly Sales Forecaster"
APP_ICON   = "📈"
VERSION    = "3.0.0"
MODEL_PATH = "model.pkl"
FEATURES   = [
    'lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_12', 'lag_26', 'lag_52',
    'ma_4', 'ma_12', 'std_4', 'weekofyear', 'month', 'year'
]

# ============================================================================
# PAGE CONFIG — must be first Streamlit command
# ============================================================================

st.set_page_config(
    page_title="Weekly Sales Forecaster",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL — cached so it only loads once at startup
# ============================================================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

# ============================================================================
# LOAD LLM CLIENT — cached
# ============================================================================

@st.cache_resource
def load_llm():
    try:
        from src.llm_client import call_llm, build_sales_prompt
        return call_llm, build_sales_prompt, None
    except Exception as e:
        return None, None, str(e)

call_llm, build_sales_prompt, llm_error = load_llm()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_prediction(
    lag_1, lag_2, lag_4, lag_8, lag_12, lag_26, lag_52,
    ma_4, ma_12, std_4, weekofyear, month, year
) -> dict:
    """
    Make prediction using the loaded model.
    Model trained on log1p(y) — expm1 converts back to dollar scale.
    """
    try:
        import time
        start    = time.time()
        features = pd.DataFrame([{
            "lag_1": lag_1, "lag_2": lag_2, "lag_4": lag_4,
            "lag_8": lag_8, "lag_12": lag_12, "lag_26": lag_26,
            "lag_52": lag_52, "ma_4": ma_4, "ma_12": ma_12,
            "std_4": std_4, "weekofyear": weekofyear,
            "month": month, "year": year
        }])
        log_pred   = float(model.predict(features)[0])
        prediction = float(np.expm1(log_pred))
        elapsed_ms = round((time.time() - start) * 1000, 2)
        lower      = prediction * 0.90
        upper      = prediction * 1.10
        confidence = f"USD {lower:,.0f} - USD {upper:,.0f} (+-10% interval)"
        return {
            "success":          True,
            "prediction":       prediction,
            "confidence":       confidence,
            "response_time_ms": elapsed_ms,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_signal(pct_change: float) -> str:
    """Return demand signal based on percentage change."""
    if abs(pct_change) <= 5:
        return "Stable"
    elif pct_change > 5:
        return "Higher demand"
    return "Lower demand"

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("📈 Weekly Sales Forecaster")
st.sidebar.markdown("**Team Dany | AIE1014**")
st.sidebar.divider()

if model is not None:
    st.sidebar.success("✅ Model Loaded")
else:
    st.sidebar.error(f"❌ Model Error: {model_error}")

if call_llm is not None:
    st.sidebar.success("✅ LLM Ready")
else:
    st.sidebar.warning("⚠️ LLM Unavailable")

st.sidebar.divider()

st.sidebar.markdown("### 📊 Model Info")
st.sidebar.write("**Model:** RandomForestRegressor v2")
st.sidebar.write("**Training:** Per-store features")
st.sidebar.write("**Features:** 13")
st.sidebar.write("**R2 Score:** 0.9812")
st.sidebar.write("**RMSE:** $73,473")
st.sidebar.write("**MAE:** $47,281")
st.sidebar.write("**Avg Response:** ~10ms")

st.sidebar.divider()
st.sidebar.markdown("**Stakeholder:** Retail Business Manager")
st.sidebar.caption(f"Version {VERSION} | AIE1014 Capstone")

# ============================================================================
# MAIN TITLE
# ============================================================================

st.title("📈 Weekly Sales Forecaster — Team Dany")
st.write("Predict next week's retail sales using historical lag features and moving averages.")
st.divider()

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction", "📊 Dashboard", "📋 History", "💬 AI Advisor"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📝 Enter Your Sales Data")

    if model is None:
        st.error("Model failed to load. Please contact support.")
        st.stop()

    now = datetime.now()

    with st.form("prediction_form"):
        st.markdown("#### Lag Features")
        col1, col2 = st.columns(2)
        with col1:
            lag_1  = st.number_input("Lag 1 — Previous week ($)",  min_value=0.0, value=1500000.0)
            lag_4  = st.number_input("Lag 4 — 4 weeks ago ($)",    min_value=0.0, value=1450000.0)
            lag_12 = st.number_input("Lag 12 — 12 weeks ago ($)",  min_value=0.0, value=1400000.0)
            lag_52 = st.number_input("Lag 52 — 1 year ago ($)",    min_value=0.0, value=1380000.0)
        with col2:
            lag_2  = st.number_input("Lag 2 — 2 weeks ago ($)",    min_value=0.0, value=1480000.0)
            lag_8  = st.number_input("Lag 8 — 8 weeks ago ($)",    min_value=0.0, value=1430000.0)
            lag_26 = st.number_input("Lag 26 — 26 weeks ago ($)",  min_value=0.0, value=1390000.0)

        st.markdown("#### Moving Averages & Volatility")
        col3, col4, col5 = st.columns(3)
        with col3:
            ma_4  = st.number_input("MA 4 weeks ($)",  min_value=0.0, value=1465000.0)
        with col4:
            ma_12 = st.number_input("MA 12 weeks ($)", min_value=0.0, value=1430000.0)
        with col5:
            std_4 = st.number_input("Std Dev 4 weeks", min_value=0.0, value=25000.0)

        st.markdown("#### Date Features (Auto-filled)")
        col6, col7, col8 = st.columns(3)
        with col6:
            weekofyear = st.number_input("Week of Year", min_value=1,    max_value=52,   value=int(now.isocalendar()[1]))
        with col7:
            month      = st.number_input("Month",        min_value=1,    max_value=12,   value=now.month)
        with col8:
            year       = st.number_input("Year",         min_value=2000, max_value=2100, value=now.year)

        submitted = st.form_submit_button("🔮 Get Prediction", use_container_width=True)

    if submitted:
        with st.spinner("Generating sales forecast..."):
            result = make_prediction(
                lag_1, lag_2, lag_4, lag_8, lag_12, lag_26, lag_52,
                ma_4, ma_12, std_4, weekofyear, month, year
            )

        if result["success"]:
            prediction = result["prediction"]
            formatted  = f"${prediction:,.2f}"
            st.success(f"✅ Predicted Next-Week Sales: **{formatted}**")

            if lag_1 > 0:
                pct_change = ((prediction - lag_1) / lag_1) * 100
                direction  = "above" if pct_change >= 0 else "below"
                abs_change = abs(pct_change)

                if abs_change <= 5:
                    st.success(f"🟢 Stable week ahead — Forecast is {abs_change:.1f}% {direction} last week.")
                elif pct_change > 5:
                    st.warning(f"🟡 Higher demand expected — Forecast is {abs_change:.1f}% above last week.")
                else:
                    st.info(f"🔵 Lower demand expected — Forecast is {abs_change:.1f}% below last week.")

                st.info(
                    f"📊 This forecast is **{abs_change:.1f}% {direction}** "
                    f"last week's sales of **${lag_1:,.2f}**. "
                    f"Use this to adjust inventory and staffing for next week."
                )

            st.markdown(f"**Prediction Interval:** {result['confidence']}")

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Forecast",      formatted)
            with col2: st.metric("Week",          f"W{weekofyear}")
            with col3: st.metric("Model",         "RF v2")
            with col4: st.metric("Response Time", f"{result['response_time_ms']} ms")

            if "history" not in st.session_state:
                st.session_state.history = []
            pct = ((prediction - lag_1) / lag_1) * 100 if lag_1 > 0 else 0
            st.session_state.history.append({
                "Timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Week":               weekofyear,
                "Month":              month,
                "Year":               year,
                "Lag 1":              lag_1,
                "Lag 52":             lag_52,
                "Prediction ($)":     round(prediction, 2),
                "% vs Last Week":     round(pct, 2),
                "Response Time (ms)": result["response_time_ms"],
            })
        else:
            st.error(f"❌ Prediction failed: {result['error']}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance Dashboard")

    # Model v1 vs v2 comparison
    st.markdown("#### 📈 Model Improvement: v1 → v2")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", "0.9812", delta="+246.9% vs v1",
                  help="v1 was 0.2829 — per-store features made the difference")
    with col2:
        st.metric("RMSE", "$73,473", delta="-$1,989,094 vs v1",
                  help="v1 was $2,062,567")
    with col3:
        st.metric("MAE", "$47,281", delta="-$1,441,305 vs v1",
                  help="v1 was $1,488,586")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Features",      "13",           help="Same 13 lag features")
    with col5:
        st.metric("Training rows", "2,160",        help="Per-store: 45 stores × ~48 weeks")
    with col6:
        st.metric("Test rows",     "1,935",        help="Chronological split Jan 2012")

    st.info(
        "**Key insight:** Training per-store instead of aggregating across all 45 stores "
        "revealed individual store patterns — this single change improved R² from 0.28 to 0.98."
    )

    st.divider()
    st.markdown("#### 🎯 Feature Importance")
    if model is not None:
        try:
            df_imp = pd.DataFrame({
                "Feature":    model.feature_names_in_.tolist(),
                "Importance": model.feature_importances_.tolist()
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                df_imp, x="Importance", y="Feature", orientation="h",
                title="Feature Importance — RandomForest v2 (per-store)",
                color="Importance", color_continuous_scale="blues",
                text="Importance"
            )
            fig_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_imp.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.info("Feature importance chart unavailable.")

    st.divider()
    st.markdown("#### 📈 Prediction History Chart")
    if "history" in st.session_state and len(st.session_state.history) > 0:
        df_hist  = pd.DataFrame(st.session_state.history)
        fig_hist = px.line(
            df_hist, x="Timestamp", y="Prediction ($)",
            title="Sales Predictions Over Time",
            markers=True, line_shape="spline"
        )
        fig_hist.update_traces(line_color="#1f77b4", marker_size=8)
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Make predictions in the Prediction tab to see the chart here.")

    st.divider()
    st.markdown("#### ⚡ Response Time History")
    if "history" in st.session_state and len(st.session_state.history) > 0:
        df_hist = pd.DataFrame(st.session_state.history)
        fig_rt  = px.bar(
            df_hist, x="Timestamp", y="Response Time (ms)",
            title="Model Inference Time per Prediction (ms)",
            color="Response Time (ms)", color_continuous_scale="greens"
        )
        fig_rt.add_hline(y=100, line_dash="dash", line_color="orange",
                         annotation_text="Target < 100ms")
        fig_rt.update_layout(height=350)
        st.plotly_chart(fig_rt, use_container_width=True)
    else:
        st.info("Make predictions to see response time chart.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📋 Prediction History")

    if "history" in st.session_state and len(st.session_state.history) > 0:
        df_hist = pd.DataFrame(st.session_state.history)

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Predictions", len(df_hist))
        with col2: st.metric("Avg Forecast", f"${df_hist['Prediction ($)'].mean():,.2f}")
        with col3: st.metric("Max Forecast", f"${df_hist['Prediction ($)'].max():,.2f}")
        with col4: st.metric("Min Forecast", f"${df_hist['Prediction ($)'].min():,.2f}")

        st.dataframe(df_hist, use_container_width=True)

        csv = df_hist.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name="predictions_history.csv",
            mime="text/csv"
        )
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Go to the Prediction tab to get started!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 💬 AI Sales Advisor")
    st.write("Get AI-powered explanations and business recommendations for your forecast.")

    if call_llm is None:
        st.error(f"LLM not available: {llm_error}")
        st.info("Add your OPENAI_API_KEY to Streamlit secrets to enable AI features.")
    else:
        # ── Section 1: Explain last prediction ────────────────────────────
        st.markdown("#### 🤖 Explain My Last Prediction")

        if "history" in st.session_state and len(st.session_state.history) > 0:
            last = st.session_state.history[-1]
            pct  = last.get("% vs Last Week", 0)
            sig  = get_signal(pct)

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Last Forecast", f"${last['Prediction ($)']:,.2f}")
            with col2: st.metric("Week",          f"W{last['Week']}")
            with col3: st.metric("vs Last Week",  f"{pct:+.1f}%")

            if st.button("🤖 Generate AI Business Analysis", use_container_width=True):
                with st.spinner("AI is analyzing your forecast..."):
                    try:
                        prompt      = build_sales_prompt(
                            prediction = last["Prediction ($)"],
                            pct_change = pct,
                            signal     = sig,
                            lag_1      = last["Lag 1"],
                            weekofyear = last["Week"],
                            month      = last["Month"],
                        )
                        explanation = call_llm(prompt)
                        st.success("✅ AI Business Analysis")
                        st.write(explanation)
                    except Exception as e:
                        st.error(f"LLM error: {e}")
        else:
            st.info("Make a prediction in the Prediction tab first, then come back here for AI analysis.")

        st.divider()

        # ── Section 2: Ask anything ────────────────────────────────────────
        st.markdown("#### ❓ Ask the AI Anything About Sales")
        question = st.text_input(
            "Your question:",
            placeholder="e.g. What does a 15% sales increase mean for my inventory and staffing?"
        )

        col1, col2 = st.columns(2)
        with col1:
            ref_sales = st.number_input("Reference sales ($)", value=1500000.0, step=50000.0)
        with col2:
            ref_pct   = st.number_input("Expected % change",   value=10.0,      step=1.0)

        if st.button("💬 Ask AI", use_container_width=True) and question:
            with st.spinner("Thinking..."):
                try:
                    sig    = get_signal(ref_pct)
                    prompt = f"""You are an expert retail business analyst advising a Retail Business Manager.

Question: {question}

Context:
- Current sales level: ${ref_sales:,.0f}
- Expected % change: {ref_pct:+.1f}% ({sig})

Answer in 3-5 sentences. Be practical, specific, and use dollar amounts where relevant."""
                    answer = call_llm(prompt)
                    st.success("✅ AI Response")
                    st.write(answer)
                except Exception as e:
                    st.error(f"LLM error: {e}")

        st.divider()

        # ── Section 3: Quick recommendations ──────────────────────────────
        st.markdown("#### ⚡ Quick AI Recommendations")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📦 Inventory Advice", use_container_width=True):
                if "history" in st.session_state and st.session_state.history:
                    last = st.session_state.history[-1]
                    pct  = last.get("% vs Last Week", 0)
                    with st.spinner("Generating..."):
                        try:
                            prompt = f"""As a retail analyst, give 3 specific inventory recommendations for a store expecting {pct:+.1f}% change in sales next week (forecast: ${last['Prediction ($)']:,.0f}). Be brief and actionable."""
                            st.info(call_llm(prompt))
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.warning("Make a prediction first.")

        with col2:
            if st.button("👥 Staffing Advice", use_container_width=True):
                if "history" in st.session_state and st.session_state.history:
                    last = st.session_state.history[-1]
                    pct  = last.get("% vs Last Week", 0)
                    with st.spinner("Generating..."):
                        try:
                            prompt = f"""As a retail analyst, give 3 specific staffing recommendations for a store expecting {pct:+.1f}% change in sales next week (forecast: ${last['Prediction ($)']:,.0f}). Be brief and actionable."""
                            st.info(call_llm(prompt))
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.warning("Make a prediction first.")

        with col3:
            if st.button("⚠️ Risk Assessment", use_container_width=True):
                if "history" in st.session_state and st.session_state.history:
                    last = st.session_state.history[-1]
                    pct  = last.get("% vs Last Week", 0)
                    with st.spinner("Generating..."):
                        try:
                            prompt = f"""As a retail analyst, identify the top 3 risks for a store forecasting {pct:+.1f}% sales change next week (forecast: ${last['Prediction ($)']:,.0f}). Be brief and specific."""
                            st.warning(call_llm(prompt))
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.warning("Make a prediction first.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
col1, col2, col3 = st.columns(3)
with col1: st.caption("AIE1014 Capstone | AIE1017 LLM Integration")
with col2: st.caption(f"Version {VERSION} | Team Dany")
with col3: st.caption("Stakeholder: Retail Business Manager")