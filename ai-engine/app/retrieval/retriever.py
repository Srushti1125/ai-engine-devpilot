"""
Step 5 — Retriever (hybrid: MMR + metadata filtering).

- MMR (Maximal Marginal Relevance) avoids returning five near-duplicate
  chunks when generating things like test cases, where you want
  *positive + negative + edge* coverage, not five variations of the same
  requirement sentence.
- The metadata `filter` scopes retrieval to one project (and optionally one
  doc type) — critical once multiple projects share the same pgvector
  table/collection.
"""
from app.retrieval.vectorstore import get_vectorstore

DEFAULT_K = 6
DEFAULT_FETCH_K = 20
DEFAULT_LAMBDA_MULT = 0.5  # 0 = max diversity, 1 = max relevance


def get_retriever(
    project_id: str,
    doc_type: str | None = None,
    k: int = DEFAULT_K,
    fetch_k: int = DEFAULT_FETCH_K,
    lambda_mult: float = DEFAULT_LAMBDA_MULT,
):
    vs = get_vectorstore()
    filt: dict = {"project_id": project_id}
    if doc_type:
        filt["doc_type"] = doc_type

    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "filter": filt,
        },
    )
