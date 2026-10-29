from pydantic import BaseModel, Field


class SprintItem(BaseModel):
    story_title: str
    story_points: int
    effort_estimate_hours: float
    dependencies: list[str] = []
    sprint: int


class SprintPlan(BaseModel):
    items: list[SprintItem]
    assumptions: list[str] = Field(description="Inferred assumptions about estimates, team capacity, or dependencies")
    critical_path: list[str] = []
    assumed_velocity_points_per_sprint: int = 0
