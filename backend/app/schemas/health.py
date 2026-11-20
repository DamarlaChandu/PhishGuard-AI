"""Response schemas for the health-check endpoint."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Status of the backend service and its database connection."""

    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str
    database: Literal["connected", "unavailable"]
