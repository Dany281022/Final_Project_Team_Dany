# src/llm_client.py
"""
LLM client with automatic fallback: OpenAI → Ollama.
Same adapter pattern used across all projects.
"""
import os
import httpx
from openai import OpenAI, APIConnectionError, AuthenticationError

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2:latest")


def _call_openai(prompt: str) -> str:
    """Call OpenAI GPT with the given prompt."""
    client   = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model       = OPENAI_MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.4,
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    """Call a local Ollama model via its REST API."""
    url     = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
    }
    response = httpx.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


def call_llm(prompt: str) -> str:
    """
    Single entry point for all LLM calls.
    Tries OpenAI first, falls back to Ollama on failure.
    """
    if OPENAI_API_KEY:
        try:
            print("[LLM] Using OpenAI...")
            return _call_openai(prompt)
        except (APIConnectionError, AuthenticationError) as e:
            print(f"[LLM] OpenAI failed ({e}), falling back to Ollama...")
        except Exception as e:
            print(f"[LLM] Unexpected error ({e}), falling back to Ollama...")

    try:
        print(f"[LLM] Using Ollama ({OLLAMA_MODEL})...")
        return _call_ollama(prompt)
    except Exception as e:
        raise RuntimeError(f"All LLM providers failed. Last error: {e}")


def build_sales_prompt(
    prediction: float,
    pct_change: float,
    signal:     str,
    lag_1:      float,
    weekofyear: int,
    month:      int,
) -> str:
    """
    Build a structured prompt for sales forecast explanation
    and business recommendations.
    """
    direction = "above" if pct_change >= 0 else "below"
    return f"""You are an expert retail business analyst advising a Retail Business Manager.

SALES FORECAST DATA:
- Predicted next-week sales: ${prediction:,.2f}
- Last week's sales (lag_1): ${lag_1:,.2f}
- Change vs last week: {pct_change:+.1f}% ({direction})
- Demand signal: {signal}
- Week of year: {weekofyear} | Month: {month}

Please provide a concise business advisory (4-6 sentences) covering:
1. What this forecast means for the business
2. Why sales may be trending this way (seasonal patterns, momentum)
3. Specific inventory recommendation (increase/decrease/maintain)
4. Specific staffing recommendation
5. One key risk to watch for

Be direct, practical, and professional. Use dollar amounts where relevant."""