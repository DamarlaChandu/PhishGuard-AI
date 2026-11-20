# PhishGuard-AI

PhishGuard-AI is planned as a real-time phishing and malicious URL detection
system with a cybersecurity dashboard and a Chrome Manifest V3 browser
extension. The intended stack is Python 3.11, FastAPI, React with Vite,
Tailwind CSS, Random Forest, XGBoost, SHAP, VirusTotal, and SQLite.

> **Project status:** Initial structure only. No application code, trained
> models, datasets, API integrations, or detection results have been created.

## Project structure

```text
PhishGuard-AI/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/
│   │   ├── core/
│   │   ├── db/
│   │   ├── integrations/
│   │   ├── ml/
│   │   │   ├── explainability/
│   │   │   ├── features/
│   │   │   ├── inference/
│   │   │   └── training/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── services/
│   └── tests/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── api/
│   ├── architecture/
│   ├── ml/
│   └── security/
├── extension/
│   ├── public/icons/
│   └── src/
│       ├── background/
│       ├── content/
│       ├── options/
│       └── popup/
├── frontend/
│   ├── public/
│   └── src/
│       ├── assets/
│       ├── components/
│       │   ├── common/
│       │   └── dashboard/
│       ├── hooks/
│       ├── layouts/
│       ├── pages/
│       ├── services/
│       └── types/
├── scripts/
└── tests/
    ├── e2e/
    └── integration/
```

## Folder purposes

- `backend/` — Python API and server-side application.
- `backend/app/` — Backend application package.
- `backend/app/api/` — HTTP API routing.
- `backend/app/api/v1/` — Version 1 API routes.
- `backend/app/api/v1/endpoints/` — Individual endpoint modules.
- `backend/app/core/` — Shared configuration, settings, and core utilities.
- `backend/app/db/` — Database connection, session, and persistence setup.
- `backend/app/integrations/` — External service clients, including the planned
  VirusTotal integration.
- `backend/app/ml/` — Machine-learning pipeline and model-related code.
- `backend/app/ml/explainability/` — SHAP explanations and related logic.
- `backend/app/ml/features/` — URL feature extraction and transformation.
- `backend/app/ml/inference/` — Loading and running trained models for scans.
- `backend/app/ml/training/` — Dataset preparation, training, and evaluation.
- `backend/app/models/` — Backend data and persistence model definitions.
- `backend/app/schemas/` — API request and response schemas.
- `backend/app/services/` — Application and detection orchestration logic.
- `backend/tests/` — Backend unit and component tests.
- `data/` — Local data workspace; datasets will be added intentionally later.
- `data/external/` — External reference data or artifacts that are not model
  training inputs.
- `data/interim/` — Temporary outputs from data preparation.
- `data/processed/` — Prepared datasets ready for model workflows.
- `data/raw/` — Original datasets retained unchanged, subject to their licenses.
- `docs/` — Project design and engineering documentation.
- `docs/api/` — API contracts and usage documentation.
- `docs/architecture/` — System architecture and design decisions.
- `docs/ml/` — Dataset, feature, training, evaluation, and model documentation.
- `docs/security/` — Security assumptions, controls, and review notes.
- `extension/` — Chrome Manifest V3 browser extension.
- `extension/public/icons/` — Extension icon assets.
- `extension/src/` — Extension source code.
- `extension/src/background/` — Background service worker responsibilities.
- `extension/src/content/` — Page-context scripts and browser-page interaction.
- `extension/src/options/` — Extension settings interface.
- `extension/src/popup/` — Extension toolbar popup interface.
- `frontend/` — React and Vite dashboard.
- `frontend/public/` — Static assets copied as-is by the frontend build.
- `frontend/src/` — Dashboard source code.
- `frontend/src/assets/` — Imported images, styles, and other source assets.
- `frontend/src/components/` — Reusable interface components.
- `frontend/src/components/common/` — Dashboard-wide shared UI components.
- `frontend/src/components/dashboard/` — Dashboard-specific UI components.
- `frontend/src/hooks/` — Reusable React hooks.
- `frontend/src/layouts/` — Page layout components.
- `frontend/src/pages/` — Top-level dashboard pages.
- `frontend/src/services/` — API clients and frontend service logic.
- `frontend/src/types/` — Shared TypeScript type definitions.
- `scripts/` — Repeatable development, data, and maintenance scripts.
- `tests/` — Cross-component test suites.
- `tests/e2e/` — End-to-end system tests.
- `tests/integration/` — Integration tests spanning application components.

## Development scope

The initial delivery is intended to fit within four days. The project will use
real, appropriately licensed datasets and evaluated models; dataset sources,
provenance, and limitations should be documented before training. External API
credentials must be supplied through local environment configuration and must
not be committed.

No containers are planned. Dependencies and runnable setup instructions will
be added alongside the application in subsequent steps.
