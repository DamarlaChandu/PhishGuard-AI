# PhishGuard AI — Backend

FastAPI backend foundation for PhishGuard AI. This stage provides the
application skeleton, environment-based configuration, SQLite connectivity via
SQLAlchemy, and a health-check endpoint. URL scanning and ML detection are not
implemented yet.

## Requirements

- Python 3.10 or newer (3.11 recommended)
- Windows Command Prompt (CMD) for the commands below

## Structure

```text
backend/
├── app/
│   ├── api/v1/
│   │   ├── endpoints/health.py   # GET /api/v1/health
│   │   └── router.py             # Aggregates v1 routers
│   ├── core/config.py            # Settings loaded from .env
│   ├── db/
│   │   ├── base.py               # SQLAlchemy declarative base
│   │   └── session.py            # Engine, session factory, get_db dependency
│   ├── schemas/health.py         # Health response schema
│   └── main.py                   # FastAPI application entry point
├── tests/
│   ├── conftest.py               # Test client using an in-memory database
│   └── test_health.py
├── .env.example
├── pytest.ini
└── requirements.txt
```

## Configuration

Settings are read from environment variables, then from `backend/.env`.
Copy the example file and edit as needed:

```bat
copy .env.example .env
```

| Variable        | Default                     | Description                         |
|-----------------|-----------------------------|-------------------------------------|
| `APP_NAME`      | `PhishGuard AI Backend`     | Service name shown in API metadata  |
| `APP_VERSION`   | `0.1.0`                     | Service version                     |
| `ENVIRONMENT`   | `development`               | Deployment environment label        |
| `DEBUG`         | `false`                     | FastAPI debug mode and SQL echo     |
| `API_V1_PREFIX` | `/api/v1`                   | Route prefix for version 1 API      |
| `DATABASE_URL`  | `sqlite:///./phishguard.db` | SQLAlchemy database URL             |

`.env` is git-ignored; never commit real secrets.

## Setup and run (Windows CMD)

Run all commands from the `backend` directory.

```bat
cd backend
python -m venv venv
venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

The API is then available at <http://127.0.0.1:8000>, with interactive docs at
<http://127.0.0.1:8000/docs>.

## Health check

```bat
curl http://127.0.0.1:8000/api/v1/health
```

Example response:

```json
{"status":"ok","service":"PhishGuard AI Backend","version":"0.1.0","environment":"development","database":"connected"}
```

The endpoint runs `SELECT 1` against the database. If the database is
unreachable it returns HTTP 503 with `"status":"degraded"` and
`"database":"unavailable"`.

## Tests

```bat
python -m pytest
```

Tests use an in-memory SQLite database and do not touch `phishguard.db`.

## Public Suffix List domain extraction

URL feature extraction and training-group assignment share
`app.ml.features.domain.extract_domain_info`, backed by `tldextract==5.4.0`.
The pinned package includes a Public Suffix List snapshot. The extractor is
configured with `suffix_list_urls=()` and `cache_dir=None`, so it uses that
bundled snapshot without fetching suffix data or relying on a local cache at
runtime.

Private PSL suffixes are enabled intentionally. For example,
`tenant.blogspot.com` is treated as its own registrable domain rather than
being grouped with all other Blogspot tenants under `blogspot.com`. Update the
pin only alongside review of the bundled PSL snapshot and the domain extraction
tests.
