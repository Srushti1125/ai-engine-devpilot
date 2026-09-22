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
from langchain_postgres import PGVector

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
def get_vectorstore() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=settings.DATABASE_URL,
        use_jsonb=True,  # store metadata as jsonb -> filterable by project_id/doc_type
    )


def index_chunks(chunks) -> int:
    """Embed and upsert a batch of chunked Documents. Returns count indexed."""
    if not chunks:
        return 0
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return len(chunks)
