"""Health-check endpoint."""

from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.db.session import check_database_connection
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Report backend and database status",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
def health_check(response: Response) -> HealthResponse:
    """Return 200 when the API and database are reachable, 503 otherwise."""
    settings = get_settings()
    database_ok = check_database_connection()

    if not database_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if database_ok else "degraded",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        database="connected" if database_ok else "unavailable",
    )
