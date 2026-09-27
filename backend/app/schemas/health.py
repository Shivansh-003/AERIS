from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Overall system health status")
    project: str = Field(default="AERIS", description="Application name")
    full_name: str = Field(
        default="Adaptive Environmental Risk & Intelligence System",
        description="Full application description",
    )
    version: str = Field(default="1.0.0", description="API version")
    environment: str = Field(
        default="development", description="Current runtime environment"
    )
    uptime_seconds: float = Field(
        ..., description="Seconds since application startup"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Current server UTC timestamp",
    )
    device: str = Field(
        default="cpu", description="Active computational execution device"
    )
    models_loaded: List[str] = Field(
        default_factory=list,
        description="List of active model identifiers loaded into memory",
    )
    dataset_version: Optional[str] = Field(
        default="city_day_v1.0",
        description="Associated dataset version identifier",
    )
