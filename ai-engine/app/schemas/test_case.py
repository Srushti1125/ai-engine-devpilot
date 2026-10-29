from pydantic import BaseModel, Field


class TestCase(BaseModel):
    id: str
    title: str
    category: str = Field(description="positive | negative | boundary | security")
    steps: list[str]
    expected_result: str
    traceability: str = Field(description="Requirement or acceptance criterion this traces back to")


class TestCaseList(BaseModel):
    cases: list[TestCase]
    assumptions: list[str] = Field(description="Inferred assumptions used; return an empty list when none were needed")
