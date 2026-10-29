from pydantic import BaseModel, Field


class AcceptanceCriterion(BaseModel):
    given: str
    when: str
    then: str
    case_type: str = Field(description="positive | negative | edge")


class AcceptanceCriteriaList(BaseModel):
    user_story_title: str
    criteria: list[AcceptanceCriterion]
    assumptions: list[str] = Field(description="Inferred assumptions used; return an empty list when none were needed")
