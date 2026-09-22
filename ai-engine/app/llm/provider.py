"""
Step 7 — Model-agnostic LLM layer.

Wraps providers behind one interface so swapping models is a config change
(env var), not a code change — even though we default to Gemini's free
tier now, this keeps the door open for a paid fallback (or a different
provider entirely) later without touching pipeline/chain.py.

Why Gemini free tier:
  - Hosted API (generativelanguage.googleapis.com), NOT local/on-device —
    satisfies the "not locally hosted" constraint.
  - $0 per-token, no billing account required to start.
  - Gemini 3.x models offer up to a 1M-token context window — the largest
    of any free API tier — which matters directly for RAG, where retrieved
    chunks get stuffed into the prompt alongside the requirement text. This
    removes the "limited tokens" ceiling that a smaller free-tier context
    window (e.g. 8K-32K) would otherwise impose on chunk count/size.
  - Shares one GOOGLE_API_KEY with embeddings (Step 3) -> one provider
    account for the whole team.

Caveats to flag to the team (see .env.example / README):
  - Free-tier prompts may be used by Google for product improvement
    (no opt-out outside the EU/UK/EEA) — don't send anything sensitive
    through the free tier without checking with the team first.
  - Daily/per-minute quotas can change without notice. Keep DEFAULT_LLM
    and FALLBACK_LLM configurable (env vars), not hardcoded, and keep
    tracking/usage.py watching the quota (Step 10).

Fallback option if Gemini's daily cap gets tight: Mistral AI's free
"Experiment" tier (1B tokens/month, higher ceiling / lower per-minute rate,
better suited to batch ingestion than interactive demos). Wired in below
behind the same get_llm() interface.
"""
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=8)
def get_llm(model_alias: str | None = None, temperature: float = 0.2):
    """
    Returns a LangChain chat model instance for the given alias.

    Keep temperature low (0-0.3) for artifact generation — consistency and
    schema compliance matter far more than creative variation here.
    """
    model_alias = model_alias or settings.DEFAULT_LLM

    if model_alias.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_alias,
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    if model_alias.startswith("mistral"):
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=model_alias,
            temperature=temperature,
            api_key=settings.MISTRAL_API_KEY,
        )

    if model_alias.startswith("groq"):
        from langchain_groq import ChatGroq
        # Strip "groq-" prefix to get the actual Groq model name
        groq_model = model_alias.removeprefix("groq-")
        return ChatGroq(
            model=groq_model,
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
        )

    # Add more branches here later (OpenAI, Anthropic, etc.) without
    # touching any call site — they all just call get_llm(alias).
    raise ValueError(f"Unrecognized model alias '{model_alias}' — no provider branch matches it.")


def get_default_llm():
    return get_llm(settings.DEFAULT_LLM)


def get_fallback_llm():
    """Used only when primary generation exhausts its retries (Step 9)."""
    return get_llm(settings.FALLBACK_LLM)
