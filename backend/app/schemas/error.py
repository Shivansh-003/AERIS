from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InvalidParam(BaseModel):
    name: str
    reason: str


class ProblemDetails(BaseModel):
    type: str = Field(
        default="about:blank",
        description="URI reference identifying the problem type",
    )
    title: str = Field(..., description="Short, human-readable summary of the problem")
    status: int = Field(..., description="HTTP status code")
    detail: Optional[str] = Field(
        None, description="Human-readable explanation specific to this occurrence"
    )
    instance: Optional[str] = Field(
        None, description="URI reference identifying the specific occurrence"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    invalid_params: Optional[List[InvalidParam]] = None
    extra: Optional[Dict[str, Any]] = None
