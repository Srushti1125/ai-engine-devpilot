"""Streamlit interface for the DevPilot AI Engine.

Run from the ``ai-engine`` directory with ``streamlit run streamlit_app.py``.
Configure GOOGLE_API_KEY and GROQ_API_KEY as environment variables or
Streamlit secrets before deploying. Community Cloud uses local Chroma storage.
"""
import os
import json
import tempfile
from pathlib import Path

import streamlit as st

# Community Cloud demo uses a local Chroma index, so it doesn't need a
# PostgreSQL DATABASE_URL secret. The API keeps pgvector as its default.
os.environ.setdefault("VECTOR_STORE_BACKEND", "chroma")

# Streamlit Community Cloud secrets are exposed as st.secrets. Populate the
# environment before importing app.config, which reads settings at import time.
for _key in (
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "DATABASE_URL",
    "DEFAULT_LLM",
    "FALLBACK_LLM",
    "MISTRAL_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
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
st.caption("Upload the project requirements, then describe what you want this artifact to account for.")


def _text(value) -> str:
    return str(value) if value not in (None, "") else "—"


def _render_artifact(artifact_type: str, data: dict) -> None:
    """Render generated schema fields as readable sections and cards."""
    if artifact_type == "user_story":
        for i, story in enumerate(data.get("stories", []), 1):
            with st.container(border=True):
                st.markdown(f"### {i}. {_text(story.get('title'))}")
                st.caption(f"{_text(story.get('priority'))} priority · {_text(story.get('category'))}")
                st.write(_text(story.get("story")))
                if story.get("dependencies"):
                    st.markdown("**Dependencies**")
                    for dependency in story["dependencies"]:
                        st.markdown(f"- {dependency}")

    elif artifact_type == "acceptance_criteria":
        st.markdown(f"#### User story: {_text(data.get('user_story_title'))}")
        for i, criterion in enumerate(data.get("criteria", []), 1):
            with st.container(border=True):
                st.markdown(f"**Criterion {i} · {_text(criterion.get('case_type'))}**")
                st.markdown(f"**Given:** {_text(criterion.get('given'))}")
                st.markdown(f"**When:** {_text(criterion.get('when'))}")
                st.markdown(f"**Then:** {_text(criterion.get('then'))}")

    elif artifact_type == "test_case":
        for case in data.get("cases", []):
            with st.container(border=True):
                st.markdown(f"### {_text(case.get('id'))} · {_text(case.get('title'))}")
                st.caption(_text(case.get("category")).title())
                st.markdown("**Steps**")
                for i, step in enumerate(case.get("steps", []), 1):
                    st.markdown(f"{i}. {step}")
                st.markdown(f"**Expected result:** {_text(case.get('expected_result'))}")
                st.markdown(f"**Traces to:** {_text(case.get('traceability'))}")

    elif artifact_type == "api_spec":
        for endpoint in data.get("endpoints", []):
            with st.container(border=True):
                st.markdown(f"### `{_text(endpoint.get('method')).upper()}` `{_text(endpoint.get('path'))}`")
                st.write(_text(endpoint.get("description")))
                for label, fields in (("Request", endpoint.get("request_schema", [])), ("Response", endpoint.get("response_schema", []))):
                    if fields:
                        st.markdown(f"**{label} fields**")
                        st.dataframe(
                            [{"Name": f.get("name"), "Type": f.get("type"), "Required": f.get("required"), "Description": f.get("description")} for f in fields],
                            hide_index=True,
                            use_container_width=True,
                        )
                if endpoint.get("error_codes"):
                    st.markdown("**Error codes**")
                    for error in endpoint["error_codes"]:
                        st.markdown(f"- `{error.get('code')}` — {error.get('reason')}")

    elif artifact_type == "architecture":
        if data.get("pattern"):
            st.markdown(f"### {data['pattern']}")
        if data.get("rationale"):
            st.markdown("#### Rationale")
            st.write(data["rationale"])
        if data.get("technology_recommendations"):
            st.markdown("#### Technology recommendations")
            for technology in data["technology_recommendations"]:
                st.markdown(f"- {technology}")
        if data.get("components"):
            st.markdown("#### Components")
            for component in data["components"]:
                with st.container(border=True):
                    st.markdown(f"**{_text(component.get('name'))}** · {_text(component.get('technology'))}")
                    st.write(_text(component.get("responsibility")))
        if data.get("data_flow"):
            st.markdown("#### Data flow")
            st.write(data["data_flow"])
        if data.get("security_considerations"):
            st.markdown("#### Security considerations")
            for item in data["security_considerations"]:
                st.markdown(f"- {item}")

    elif artifact_type == "sprint_plan":
        st.markdown(f"**Assumed velocity:** {_text(data.get('assumed_velocity_points_per_sprint'))} points per sprint")
        items = data.get("items", [])
        if items:
            st.dataframe(
                [{
                    "Sprint": item.get("sprint"),
                    "Story": item.get("story_title"),
                    "Points": item.get("story_points"),
                    "Effort (hours)": item.get("effort_estimate_hours"),
                    "Dependencies": ", ".join(item.get("dependencies", [])) or "—",
                } for item in items],
                hide_index=True,
                use_container_width=True,
            )
        if data.get("critical_path"):
            st.markdown("#### Critical path")
            for i, step in enumerate(data["critical_path"], 1):
                st.markdown(f"{i}. {step}")

    else:
        # Keep new schema types readable without exposing a raw JSON block.
        for key, value in data.items():
            st.markdown(f"#### {key.replace('_', ' ').title()}")
            if isinstance(value, list):
                for item in value:
                    st.markdown(f"- {_text(item)}")
            else:
                st.write(value)

with st.sidebar:
    st.header("Project")
    project_id = st.text_input("Project ID", value="demo-project", help="Used to scope retrieval in the vector store.")
    artifact_type = st.selectbox("Artifact to generate", options=sorted(SCHEMA_REGISTRY))
    st.divider()
    st.caption("Set GOOGLE_API_KEY and GROQ_API_KEY in Community Cloud secrets. The demo uses a local vector index.")

generation_specs = st.text_area(
    "Generation specifications",
    height=240,
    placeholder="Example for a sprint plan: 4 team members, 2-week sprints, 6 working hours per member per day. Plan the work across 3 sprints.",
    help="These are extra constraints for this generation. The uploaded requirements document remains the source of project requirements.",
)

uploaded_file = st.file_uploader(
    "Project requirements document (required)",
    type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
)

generate_clicked = st.button("Generate artifact", type="primary", use_container_width=True)

if generate_clicked:
    if not project_id.strip():
        st.error("Enter a project ID.")
    elif uploaded_file is None:
        st.error("Upload the project requirements document before generating an artifact.")
    elif not generation_specs.strip():
        st.error("Enter generation specifications, such as team size or sprint duration.")
    else:
        try:
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            try:
                with st.spinner("Reading and indexing the uploaded requirements document…"):
                    documents = load_requirement_doc(temp_path, project_id.strip())
                    chunks = chunk_documents(documents)
                    count = index_chunks(chunks)
                st.success(f"Indexed {count} requirement chunk(s) for project {project_id.strip()}.")
            finally:
                Path(temp_path).unlink(missing_ok=True)

            generation_prompt = (
                f"Use the uploaded project requirements as the source of truth. "
                f"Generate a {artifact_type.replace('_', ' ')}.\n\n"
                f"Additional generation specifications from the user:\n{generation_specs.strip()}"
            )

            with st.spinner(f"Generating {artifact_type.replace('_', ' ')}…"):
                from app.schemas.registry import get_schema

                response = run_generation(
                    artifact_type=artifact_type,
                    project_id=project_id.strip(),
                    requirement_text=generation_prompt,
                    schema_cls=get_schema(artifact_type),
                )

            st.subheader(artifact_type.replace("_", " ").title())
            _render_artifact(artifact_type, response["output"])
            with st.expander("Generation details"):
                st.write({
                    "project_id": response["project_id"],
                    "prompt_version": response["prompt_version"],
                    "usage": response["usage"],
                })
            st.download_button(
                "Download JSON",
                data=json.dumps(response["output"], indent=2, ensure_ascii=False),
                file_name=f"{artifact_type}_{project_id.strip()}.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"Could not generate the artifact: {exc}")
