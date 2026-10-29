"""Streamlit interface for the DevPilot AI Engine.

Run from the ``ai-engine`` directory with ``streamlit run streamlit_app.py``.
Configure GOOGLE_API_KEY and DATABASE_URL as environment variables or
Streamlit secrets before deploying.
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud secrets are exposed as st.secrets. Populate the
# environment before importing app.config, which reads settings at import time.
for _key in (
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "DATABASE_URL",
    "DEFAULT_LLM",
    "FALLBACK_LLM",
    "MISTRAL_API_KEY",
):
    if _key not in os.environ and _key in st.secrets:
        os.environ[_key] = str(st.secrets[_key])

from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_requirement_doc
from app.ingestion.splitters import chunk_documents
from app.pipeline.chain import run_generation
from app.retrieval.vectorstore import index_chunks
from app.schemas.registry import SCHEMA_REGISTRY


st.set_page_config(page_title="DevPilot AI Engine", page_icon="🚀", layout="wide")
st.title("DevPilot AI Engine")
st.caption("Generate structured project artifacts from requirements, with optional project context.")

with st.sidebar:
    st.header("Project")
    project_id = st.text_input("Project ID", value="demo-project", help="Used to scope retrieval in the vector store.")
    artifact_type = st.selectbox("Artifact to generate", options=sorted(SCHEMA_REGISTRY))
    st.divider()
    st.caption("Set GOOGLE_API_KEY and DATABASE_URL in your deployment environment or Streamlit secrets.")

requirement_text = st.text_area(
    "Requirements",
    height=240,
    placeholder="Describe the feature or project requirements…",
)

uploaded_file = st.file_uploader(
    "Optional: add a requirements document to this project's retrieval context",
    type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
)

generate_clicked = st.button("Generate artifact", type="primary", use_container_width=True)

if generate_clicked:
    if not project_id.strip():
        st.error("Enter a project ID.")
    elif not requirement_text.strip():
        st.error("Enter requirements to generate an artifact.")
    else:
        try:
            if uploaded_file is not None:
                suffix = Path(uploaded_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                try:
                    with st.spinner("Reading and indexing the uploaded document…"):
                        documents = load_requirement_doc(temp_path, project_id.strip())
                        chunks = chunk_documents(documents)
                        count = index_chunks(chunks)
                    st.success(f"Indexed {count} document chunk(s) for project {project_id.strip()}.")
                finally:
                    Path(temp_path).unlink(missing_ok=True)

            with st.spinner(f"Generating {artifact_type.replace('_', ' ')}…"):
                from app.schemas.registry import get_schema

                response = run_generation(
                    artifact_type=artifact_type,
                    project_id=project_id.strip(),
                    requirement_text=requirement_text.strip(),
                    schema_cls=get_schema(artifact_type),
                )

            st.subheader(artifact_type.replace("_", " ").title())
            st.json(response["output"])
            with st.expander("Generation details"):
                st.write({
                    "project_id": response["project_id"],
                    "prompt_version": response["prompt_version"],
                    "usage": response["usage"],
                })
            st.download_button(
                "Download JSON",
                data=__import__("json").dumps(response["output"], indent=2),
                file_name=f"{artifact_type}_{project_id.strip()}.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"Could not generate the artifact: {exc}")
