"""
Step 11 — Full pipeline (LCEL).

Requirement -> RAG -> Prompt -> LLM -> Structured Response -> Quality Check

This is the piece that ties every earlier step together into the one
function Person 2 calls (via the /generate HTTP endpoint in main.py).
"""
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.llm.parsers import get_parser
from app.llm.provider import get_llm
from app.pipeline.quality_check import generate_with_quality_check
from app.prompts.registry import load_prompt
from app.retrieval.retriever import get_retriever


def _format_docs(docs) -> str:
    if not docs:
        return "(no matching context retrieved for this project)"
    return "\n\n".join(d.page_content for d in docs)


def build_generation_chain(
    artifact_type: str,
    project_id: str,
    schema_cls,
    model_alias: str | None = None,
    doc_type: str | None = "requirement",
    retrieval_trace: list[dict] | None = None,
):
    """
    Assembles the RAG -> Prompt -> LLM runnable for one artifact type.

    Returns (rag_chain, parser, version_id). The chain expects to be
    invoked with a dict containing at least {"requirement": ...} and
    optionally {"repair_note": ...} (quality_check.py manages the latter).
    """
    retriever = get_retriever(project_id, doc_type=doc_type)
    prompt, version_id = load_prompt(artifact_type)
    llm = get_llm(model_alias)
    parser = get_parser(schema_cls)

    def _inputs(x: dict) -> dict:
        requirement_text = x["requirement"]
        docs = retriever.invoke(requirement_text)
        if retrieval_trace is not None:
            retrieval_trace.clear()
            seen = set()
            for doc in docs:
                metadata = doc.metadata or {}
                source = metadata.get("source_name") or metadata.get("source") or "Uploaded requirement"
                key = (source, metadata.get("chunk_index"))
                if key in seen:
                    continue
                seen.add(key)
                sections = [
                    metadata[name]
                    for name in ("h1", "h2", "h3")
                    if metadata.get(name)
                ]
                retrieval_trace.append({
                    "source": source,
                    "sections": sections,
                    "chunk_index": metadata.get("chunk_index"),
                    "excerpt": doc.page_content[:1200],
                })
        return {
            "requirement": requirement_text,
            "context": _format_docs(docs),
            "repair_note": x.get("repair_note", ""),
            "format_instructions": parser.get_format_instructions(),
        }

    rag_chain = RunnableLambda(_inputs) | prompt | llm
    return rag_chain, parser, version_id


def run_generation(
    artifact_type: str,
    project_id: str,
    requirement_text: str,
    schema_cls,
    max_retries: int = 2,
) -> dict:
    """
    Executes the full pipeline for one generation request and returns a
    dict ready to hand back to Person 2 (and to log in `generation_history`
    for Person 5).
    """
    retrieval_trace: list[dict] = []
    chain, parser, version_id = build_generation_chain(
        artifact_type, project_id, schema_cls, retrieval_trace=retrieval_trace
    )

    fallback_chain = None
    if settings.FALLBACK_LLM and settings.FALLBACK_LLM != settings.DEFAULT_LLM:
        fallback_chain, _, _ = build_generation_chain(
            artifact_type,
            project_id,
            schema_cls,
            model_alias=settings.FALLBACK_LLM,
            retrieval_trace=retrieval_trace,
        )

    result, raw = generate_with_quality_check(
        chain,
        {"requirement": requirement_text},
        parser,
        max_retries=max_retries,
        fallback_chain=fallback_chain,
    )

    usage = getattr(raw, "usage_metadata", None) or {}

    return {
        "artifact_type": artifact_type,
        "project_id": project_id,
        "prompt_version": version_id,
        "output": result.model_dump(),
        "sources": retrieval_trace,
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "total_cost_usd": 0.0,  # free tier
        },
    }
