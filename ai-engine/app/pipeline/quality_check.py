"""
Step 9 — Quality check, retry & fallback.

Person 2 should never receive raw, unvalidated LLM output. This module:
  1. Validates output against the Pydantic schema for the artifact type.
  2. On failure, retries with the parser's error fed back into the prompt
     as a `repair_note` so the model can self-correct.
  3. Falls back to a secondary model if retries are exhausted.

This is also where the "format compliance" and "consistency" checks
Person 5's evaluation framework needs are implemented — e.g. re-running the
same input twice and diffing outputs for a consistency score.
"""
import json
import re

from langchain_core.exceptions import OutputParserException

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    """
    Gemini occasionally wraps JSON in ```json ... ``` fences despite being
    told not to. Strip them before handing off to the strict Pydantic
    parser rather than failing and burning a retry on a cosmetic issue.
    """
    return _CODE_FENCE_RE.sub("", text).strip()


def _extract_text(raw) -> str:
    return raw.content if hasattr(raw, "content") else str(raw)


def generate_with_quality_check(
    chain,
    inputs: dict,
    parser,
    max_retries: int = 2,
    fallback_chain=None,
):
    """
    Runs `chain` against `inputs`, validating the result against `parser`.

    inputs must NOT already contain "repair_note" — this function manages
    that key across retry attempts (defaulting to "" on the first attempt
    so the prompt template that always references {repair_note} doesn't
    break on attempt 1).

    fallback_chain: an alternate full RAG chain (same retriever + prompt,
    different LLM — see pipeline/chain.py) invoked once, after max_retries
    is exhausted on the primary chain.
    """
    last_error = None
    attempt_inputs = dict(inputs)
    attempt_inputs.setdefault("repair_note", "")

    for attempt in range(max_retries):
        try:
            raw = chain.invoke(attempt_inputs)
            text = _strip_code_fences(_extract_text(raw))
            parsed = parser.parse(text)
            return parsed, raw
        except (OutputParserException, ValueError, json.JSONDecodeError) as e:
            last_error = e
            attempt_inputs["repair_note"] = (
                f"Your previous output was invalid: {e}. "
                f"Fix it and respond with ONLY the corrected JSON, no markdown fences."
            )

    if fallback_chain is not None:
        try:
            raw = fallback_chain.invoke(attempt_inputs)
            text = _strip_code_fences(_extract_text(raw))
            parsed = parser.parse(text)
            return parsed, raw
        except (OutputParserException, ValueError, json.JSONDecodeError) as e:
            last_error = e

    raise RuntimeError(f"Generation failed after {max_retries} retries: {last_error}")


def consistency_score(output_a: dict, output_b: dict) -> float:
    """
    Very simple consistency metric for Person 5's evaluation framework:
    re-run the same input twice (same artifact_type + requirement_text) and
    diff the resulting dicts at the top level. Returns a 0.0-1.0 score:
    fraction of top-level keys whose values are identical between the two
    runs.

    This is intentionally simple/interpretable — Person 5 can layer
    embedding-similarity or LLM-judge scoring on top of this later without
    changing this function's contract (two dicts in, float out).
    """
    keys = set(output_a.keys()) | set(output_b.keys())
    if not keys:
        return 1.0
    matches = sum(1 for k in keys if output_a.get(k) == output_b.get(k))
    return matches / len(keys)


def format_compliance_score(raw_text: str, schema_cls) -> bool:
    """
    Did the raw LLM output parse cleanly against the schema without needing
    a retry? Used by Person 5 to score prompt versions on strict schema
    compliance, independent of retry-assisted success.
    """
    from app.llm.parsers import get_parser
    try:
        get_parser(schema_cls).parse(_strip_code_fences(raw_text))
        return True
    except Exception:
        return False
