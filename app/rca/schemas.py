from typing import Optional, List
from pydantic import BaseModel, Field

class DiagnoseRequest(BaseModel):
    query: str = Field(..., description="Free-text fault description")
    component: Optional[str] = Field(None, description="Filter by component (optional)")
    model: Optional[str] = Field(None, description="Filter by model within a component (optional)")
    top_k: int = Field(10, ge=1, le=50, description="Maximum number of matches to return")

class DiagnoseResponseItem(BaseModel):
    component: str
    model: Optional[str] = None
    matched_fault_description: str
    root_cause: str
    corrective_action: str
    similarity: float

class ComponentsResponse(BaseModel):
    components: List[str]

class ModelsResponse(BaseModel):
    models: List[str]