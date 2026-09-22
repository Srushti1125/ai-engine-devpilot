"""
Step 10 — Token & cost tracking.

Gemini responses carry token counts in `usage_metadata` on the returned
message rather than through a provider-specific callback (that pattern is
OpenAI-only) — pulled directly off the response in pipeline/chain.py.

Since the free tier has no per-token cost, what actually matters
operationally is request-count against Gemini's daily quota (1,500 req/day
on Flash, at time of writing — subject to change without notice, hence
DAILY_REQUEST_QUOTA being an env var, not a hardcoded constant). This is
what Person 5 needs for capacity planning during the case-study /
evaluation runs, not just a cost figure that's always $0.

Keep the `generation_history` table schema Person 2 owns provider-agnostic
(`prompt_tokens`, `completion_tokens`, `total_cost_usd`) so switching
providers later doesn't require a migration — this module's output dict
shape is designed to map 1:1 onto that table.

NOTE: the in-memory counter below resets on service restart. It's a
same-process guardrail/warning mechanism for local dev and demos, not a
durable quota tracker — Person 2's `generation_history` table (populated
from every /generate response's `usage` block) is the durable source of
truth for "how many requests today", and should be queried for anything
that needs to survive a restart or be shared across multiple container
instances.
"""
import logging
import threading
from datetime import date

from app.config import settings

logger = logging.getLogger("devpilot.ai_engine.usage")

_lock = threading.Lock()
_state = {"day": date.today(), "count": 0}


def record_request() -> dict:
    """
    Call once per /generate call. Returns the current day's running count
    and whether the configured daily quota has been exceeded, so callers
    (main.py) can log a warning at the boundary.
    """
    with _lock:
        today = date.today()
        if _state["day"] != today:
            _state["day"] = today
            _state["count"] = 0
        _state["count"] += 1
        count = _state["count"]

    quota_exceeded = count > settings.DAILY_REQUEST_QUOTA
    if quota_exceeded:
        logger.warning(
            "Gemini free-tier daily request quota likely exceeded: "
            "%d requests today (quota=%d). Consider FALLBACK_LLM or Mistral fallback.",
            count, settings.DAILY_REQUEST_QUOTA,
        )
    elif count == int(settings.DAILY_REQUEST_QUOTA * 0.9):
        logger.warning(
            "Approaching Gemini free-tier daily request quota: %d/%d requests today.",
            count, settings.DAILY_REQUEST_QUOTA,
        )

    return {"requests_today": count, "quota": settings.DAILY_REQUEST_QUOTA, "quota_exceeded": quota_exceeded}


def build_usage_record(usage_metadata: dict | None) -> dict:
    """
    Normalizes a LangChain message's usage_metadata into the provider-
    agnostic shape Person 2's generation_history table should store.
    """
    usage_metadata = usage_metadata or {}
    return {
        "prompt_tokens": usage_metadata.get("input_tokens", 0),
        "completion_tokens": usage_metadata.get("output_tokens", 0),
        "total_tokens": usage_metadata.get("total_tokens", 0),
        "total_cost_usd": 0.0,  # free tier -> track requests/tokens against quota instead
    }
