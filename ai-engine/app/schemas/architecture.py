from pydantic import BaseModel, Field


class ArchitectureComponent(BaseModel):
    name: str
    responsibility: str
    technology: str


class Architecture(BaseModel):
    assumptions: list[str] = Field(description="Inferred assumptions used; return an empty list when none were needed")
    pattern: str = ""
    rationale: str = ""
    technology_recommendations: list[str] = []
    components: list[ArchitectureComponent] = []
    data_flow: str = ""
    security_considerations: list[str] = []
