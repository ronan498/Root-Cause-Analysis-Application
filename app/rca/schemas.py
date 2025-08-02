from typing import Optional, List
from pydantic import BaseModel, Field

class DiagnoseRequest(BaseModel):
    query: str = Field(..., description="Free-text fault description")
    component: Optional[str] = Field(None, description="Filter by component (optional)")
    top_k: int = Field(3, ge=1, le=10, description="Number of matches to return")

class DiagnoseResponseItem(BaseModel):
    component: str
    matched_fault_description: str
    root_cause: str
    corrective_action: str
    similarity: float

class ComponentsResponse(BaseModel):
    components: List[str]
