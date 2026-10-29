"""
Step 1 — Document ingestion.

Handles mixed file types coming from the frontend's requirement-upload
screen (Person 1) and preserves structure as metadata so Person 4 can build
traceability (e.g. "this test case traces back to section 3, chunk 12 of
requirements_v2.docx") later.

Keep `load_requirement_doc`'s signature stable: `project_id` and `doc_type`
in metadata are what the retriever filters on (see retrieval/retriever.py)
and what Person 4 depends on downstream.

WHERE THE FILE ACTUALLY COMES FROM (not the database):
  1. Person 1's frontend uploads the raw file to Person 2's backend.
  2. Person 2's backend stores it in S3 (never on its own local disk, since
     nothing here is locally hosted / ECS containers are ephemeral) and
     calls this service's POST /ingest with either:
       - an "s3://bucket/key" URI (this service downloads it itself), or
       - a path on a filesystem the two containers share (rare — only if
         Person 2 mounts the same EFS volume).
  3. This module resolves that into a local tmp file, loads it, and tags
     every resulting Document with project_id/doc_type metadata before
     handing off to chunk_documents().
  4. The database (Postgres/pgvector) only ever stores the *embedded
     chunks*, never the original file — that's what index_chunks() in
     retrieval/vectorstore.py writes.
"""
import pathlib
import tempfile
import urllib.parse

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".pptx", ".xlsx", ".html",
}


def _load_with_unstructured(file_path: str) -> list[Document]:
    """
    Uses UnstructuredFileLoader (the classic, broadly-compatible loader
    present across most langchain-community versions) rather than the
    newer UnstructuredLoader class, which only exists in more recent
    langchain-community releases and additionally expects the
    `unstructured-client` package.
    """
    from langchain_community.document_loaders import UnstructuredFileLoader
    loader = UnstructuredFileLoader(file_path, mode="single")
    return loader.load()


def _load_plain_text(file_path: str) -> list[Document]:
    """
    Direct read for .txt/.md files — deliberately bypasses `unstructured`
    entirely for this case, since plain text needs no parsing library at
    all, and `unstructured`'s filetype detection depends on `libmagic`
    (a C library). On Windows, the package that used to bundle libmagic
    (`python-magic-bin`) has been pulled from PyPI, so relying on it here
    would be a dead end for anyone on Windows without a separately
    installed libmagic DLL. Reading the file directly avoids the problem
    altogether for the one file type that doesn't need it.
    """
    text = pathlib.Path(file_path).read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={})]


def _load_pdf_text(file_path: str) -> list[Document]:
    """Extract a PDF text layer without requiring system OCR utilities."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return [Document(page_content=text, metadata={})] if text else []


def _load_scanned_pdf_fallback(file_path: str) -> list[Document]:
    """
    Fallback for scanned/image-only PDFs that Unstructured's hi_res strategy
    can't get clean text out of. Uses AWS Textract (hosted OCR) since the
    whole stack is already on AWS (RDS) — no local OCR binaries to install
    or maintain.

    Requires AWS credentials with Textract access in the environment
    (same IAM role the service runs under on ECS/Fargate).
    """
    from langchain_community.document_loaders import AmazonTextractPDFLoader
    loader = AmazonTextractPDFLoader(file_path)
    return loader.load()


def _download_from_s3(s3_uri: str) -> str:
    """
    Downloads an s3://bucket/key URI to a local tmp file and returns the
    tmp path. Requires the service's IAM role (ECS task role — configured
    by Person 6) to have s3:GetObject on the bucket Person 2 uploads to.
    """
    import boto3

    parsed = urllib.parse.urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    suffix = "." + key.rsplit(".", 1)[-1] if "." in key else ""

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    boto3.client("s3").download_file(bucket, key, tmp.name)
    return tmp.name


def load_requirement_doc(
    file_path: str,
    project_id: str,
    doc_type: str = "requirement",
) -> list[Document]:
    """
    Load a single file and tag every resulting chunk-of-a-document with the
    metadata the rest of the pipeline depends on.

    Args:
        file_path: EITHER an "s3://bucket/key" URI (the normal case —
            Person 2's backend uploads the raw file to S3 and passes us
            the key) OR a local filesystem path reachable from this
            container (only if a shared volume is set up). This function
            downloads S3 URIs to a local tmp file transparently.
        project_id: DevPilot project this doc belongs to — used later to
            scope retrieval so multiple projects sharing one pgvector table
            never leak context into each other.
        doc_type: "requirement" | "srs" | "existing_code_doc" | etc. Lets
            the retriever narrow (e.g. only requirement docs, not old
            architecture notes) when building context for a given artifact
            type.

    Returns:
        List of langchain Documents with populated metadata, ready for
        chunk_documents().
    """
    local_path = file_path
    if file_path.startswith("s3://"):
        local_path = _download_from_s3(file_path)

    ext = local_path.lower().rsplit(".", 1)[-1]
    if f".{ext}" not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if ext in ("txt", "md"):
            docs = _load_plain_text(local_path)
        else:
            docs = _load_with_unstructured(local_path)
        if not docs or all(not d.page_content.strip() for d in docs):
            raise ValueError("Extraction produced no text")
    except Exception as parse_error:
        # Community Cloud may not provide all of Unstructured's native PDF
        # utilities. Try pypdf for ordinary PDFs, then Textract for scans.
        if ext == "pdf":
            try:
                docs = _load_pdf_text(local_path)
            except Exception:
                docs = []
            if not docs:
                try:
                    docs = _load_scanned_pdf_fallback(local_path)
                except Exception as fallback_error:
                    raise ValueError(
                        "Could not extract text from this PDF. For scanned/image-only PDFs, "
                        "configure AWS Textract credentials and permissions in Streamlit secrets. "
                        f"Details: {fallback_error}"
                    ) from parse_error
        else:
            raise

    for d in docs:
        d.metadata.update({
            "project_id": project_id,
            "doc_type": doc_type,
            # Record the ORIGINAL reference (s3:// URI or shared path), not
            # the local tmp path, so traceability metadata stays meaningful
            # after the tmp file is gone.
            "source": file_path,
        })
    return docs
