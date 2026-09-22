"""
Owned jointly with Person 4: Person 4 defines what "valid" means for a user
story artifact; this schema is the parsing contract Person 3's pipeline
parses LLM output against.
"""
from pydantic import BaseModel, Field


class UserStory(BaseModel):
    title: str
    story: str = Field(description="As a <role>, I want <goal>, so that <benefit>")
    priority: str = Field(description="High | Medium | Low")
    category: str
    dependencies: list[str] = []


class UserStoryList(BaseModel):
    stories: list[UserStory]
