"""Shared pytest fixtures."""

import os
from collections.abc import Iterator

# Use an isolated in-memory database so tests never touch the local SQLite file.
# This must be set before the application modules are imported.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ENVIRONMENT"] = "test"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a TestClient that runs the application lifespan."""
    with TestClient(app) as test_client:
        yield test_client
