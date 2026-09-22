"""
Step 8 — Structured output (Pydantic parsing).

Person 4 owns *what a valid artifact looks like* (the Pydantic schema);
this module owns *getting the LLM to reliably produce it* (format
instructions injected into the prompt + strict parsing of the response).
"""
from langchain_core.output_parsers import PydanticOutputParser


def get_parser(schema_cls):
    return PydanticOutputParser(pydantic_object=schema_cls)
