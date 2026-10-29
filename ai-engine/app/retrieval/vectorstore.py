"""
Step 3 (embeddings) + Step 4 (vector store).

Embeddings: Google Gemini's free-tier embeddings API
(`text-embedding-004`, 768 dimensions) — hosted, no credit card, and shares
the same GOOGLE_API_KEY as generation, so there's only one provider account
for the whole team to manage.

*** Dimension is locked at 768 (not 1536, which is the OpenAI default people
often assume). Tell Person 2 to size the pgvector column as vector(768). ***

Vector store: pgvector running as an extension inside the SAME shared
Postgres (AWS RDS) instance Person 2 already owns for the app DB. This
means no separate vector-DB infra to provision, and Person 2's existing
backups/migrations cover it automatically. Person 2 needs to run once,
on that RDS instance:

    CREATE EXTENSION IF NOT EXISTS vector;
"""
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

EMBEDDING_MODEL = settings.EMBEDDING_MODEL  # "models/text-embedding-004" -> 768 dims
COLLECTION_NAME = settings.COLLECTION_NAME


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    if settings.VECTOR_STORE_BACKEND == "chroma":
        from langchain_community.vectorstores import Chroma

        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )

    if settings.VECTOR_STORE_BACKEND == "pgvector":
        from langchain_postgres import PGVector

        if not settings.DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is required when VECTOR_STORE_BACKEND is 'pgvector'."
            )
        return PGVector(
            embeddings=get_embeddings(),
            collection_name=COLLECTION_NAME,
            connection=settings.DATABASE_URL,
            use_jsonb=True,
        )

    raise ValueError(
        f"Unsupported VECTOR_STORE_BACKEND '{settings.VECTOR_STORE_BACKEND}'. "
        "Choose 'pgvector' or 'chroma'."
    )


def index_chunks(chunks) -> int:
    """Embed and upsert a batch of chunked Documents. Returns count indexed."""
    if not chunks:
        return 0
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return len(chunks)


def index_document_chunks(chunks) -> tuple[int, bool]:
    """Reuse identical Chroma documents or replace older versions by filename.

    Returns (chunk_count, reused_existing). This path is used by the Streamlit
    uploader; the API's existing /ingest contract continues to use index_chunks.
    """
    if not chunks:
        return 0, False

    if settings.VECTOR_STORE_BACKEND != "chroma":
        return index_chunks(chunks), False

    vs = get_vectorstore()
    metadata = chunks[0].metadata
    project_id = metadata["project_id"]
    source_name = metadata["source_name"]
    fingerprint = metadata["document_fingerprint"]
    where = {"$and": [{"project_id": project_id}, {"source_name": source_name}]}
    existing = vs.get(where=where, include=["metadatas"])
    existing_ids = existing.get("ids", [])
    existing_metadata = existing.get("metadatas", []) or []

    if existing_ids and len(existing_metadata) == len(existing_ids) and all(
        item.get("document_fingerprint") == fingerprint
        for item in existing_metadata
    ):
        return len(existing_ids), True

    if existing_ids:
        vs.delete(ids=existing_ids)

    vs.add_documents(chunks)
    return len(chunks), False
