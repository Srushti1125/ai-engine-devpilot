"""
Step 2 — Chunking.

Section-aware first (when the doc has clean markdown-style headers, which is
common in structured SRS exports), then recursive within each section. This
keeps a requirement and its sub-bullets together instead of splitting
mid-thought, and gives the conversational assistant (Person 5 / UC-14)
section-level metadata for free when someone asks "show requirements under
section 3."
"""
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def _looks_like_markdown(text: str) -> bool:
    return any(line.strip().startswith("#") for line in text.splitlines()[:50])


def chunk_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split loaded documents into retrieval-sized chunks while preserving all
    upstream metadata (project_id, doc_type, source) and adding a stable
    chunk_index for traceability.
    """
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    all_chunks: list[Document] = []
    for doc in docs:
        base_metadata = dict(doc.metadata)

        if _looks_like_markdown(doc.page_content):
            try:
                header_sections = md_splitter.split_text(doc.page_content)
            except Exception:
                header_sections = [doc]
        else:
            header_sections = [doc]

        for section in header_sections:
            # MarkdownHeaderTextSplitter returns plain Documents with only
            # header metadata (h1/h2/h3) — merge in the original metadata.
            section_metadata = {**base_metadata, **section.metadata}
            section_doc = Document(page_content=section.page_content, metadata=section_metadata)
            sub_chunks = recursive_splitter.split_documents([section_doc])
            all_chunks.extend(sub_chunks)

    for i, c in enumerate(all_chunks):
        c.metadata["chunk_index"] = i

    return all_chunks
