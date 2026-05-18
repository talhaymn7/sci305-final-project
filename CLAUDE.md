# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server (hot-reload)
uvicorn app.main:app --reload
# API available at http://localhost:8000/api/v1

# Run all tests
pytest

# Run a single test file
pytest tests/test_ranking.py

# Run a single test by name
pytest tests/test_ranking.py::test_rank_fields_basic

# Seed sample data
python seed.py

# Apply database migrations
alembic upgrade head

# Create a new migration after model changes
alembic revision --autogenerate -m "description"
```

No formatter or linter is formally configured yet. Use 4-space indentation and match surrounding style.

## Architecture

AgriMind is a FastAPI backend for AI-powered agricultural field ranking and recommendation. The architecture has clear layers — keep logic in the correct one:

| Layer | Path | Responsibility |
|-------|------|----------------|
| Routes | `app/api/` | HTTP parsing, orchestration calls, error → HTTPException translation |
| Services | `app/services/` | Database reads/writes, return domain objects or explicit not-found |
| Engines | `app/engines/` | Deterministic business logic (scoring, ranking, explanation); no HTTP |
| AI | `app/ai/` | Provider contracts, registry, orchestration workflows, concrete providers |
| Models | `app/models/` | SQLAlchemy ORM models |
| Schemas | `app/schemas/` | Pydantic I/O contracts; no business logic |
| Config | `config/` | Static config (`scoring_weights.json`) |
| Migrations | `migrations/` | Alembic history |

**Do not put business logic in routes. Do not put DB access in engines.**

## Pluggable AI Provider System

The `app/ai/` subtree is the most architecturally distinct part of the repo. It implements a provider pattern so deterministic rules, ML models, and LLM backends can be swapped via environment variables without touching business logic.

```
app/ai/
├── contracts/          # Abstract Protocol classes defining each AI capability
├── providers/
│   ├── rule_based/     # Deterministic scoring (suitability, risk, explanation, ranking aug, extraction)
│   ├── ml/             # XGBoost yield prediction
│   ├── llm/            # OpenAI assistant for natural-language Q&A
│   └── stub/           # Test/dev stubs
├── orchestration/      # End-to-end workflows (ranking, recommendation, yield, agri_assistant)
└── registry.py         # Resolves provider from env vars at startup
```

Provider selection is via `.env`:
- `YIELD_PROVIDER`: `stub` | `ml` | `xgboost`
- `EXPLANATION_PROVIDER`: `deterministic` | `rule_based`
- `RISK_PROVIDER`: `rule_based` | `stub`
- `EXTRACTION_PROVIDER`: `manual` | `rule_based` | `stub`
- `AI_ASSISTANT_PROVIDER`: `openai`

Invalid provider IDs fail fast on startup. Use `stub` providers for local dev without external dependencies.

## Core Domain Workflows

1. **Ranking** (`POST /api/v1/rank-fields/`) — scores multiple fields for a crop using suitability, climate, economic, and yield factors; returns a ranked list with scores and explanations.
2. **Recommendation** (`GET /api/v1/recommendation/{field_id}/{crop_id}`) — single field/crop assessment with human-readable explanation.
3. **Management Plan** (`GET /api/v1/fields/{field_id}/management-plan`) — weekly irrigation and fertilizer prescriptions based on active crop cycle.
4. **Agri Assistant** (`POST /api/v1/agri-assistant/ask`) — LLM-powered Q&A grounded in ranking results (requires `OPENAI_API_KEY`).

## Scoring

Weights live in `config/scoring_weights.json`. The suitability engine scores: pH (20), soil compatibility (24), water availability (16), drainage (12), climate (20), slope (8). Minimum field area is a **blocking constraint** — undersized fields are excluded regardless of other scores.

Climate and economic data are ingested from NASA POWER and FAOSTAT respectively and fed into scoring via `app/ingestion/` and `app/engines/`.

## Data Model Key Entities

`Field` → has many `SoilTest`, `WeatherHistory`, `FieldCropCycle` → references `CropProfile`. Rankings and recommendations reference both.

## Testing

Tests use pytest with an **in-memory SQLite** database (not PostgreSQL). Fixtures are in `tests/conftest.py`: `db`, `client`, `created_field`, `created_crop`, `sample_field_data`, `sample_soil_data`.

Cover normal flow, edge cases, and failure paths. For scoring/ranking changes, add targeted business-logic tests in addition to API tests.

## Naming Conventions

- DB columns: `snake_case` with unit suffixes (`area_hectares`, `slope_percent`, `nitrogen_ppm`, `ph_level`)
- Timestamps: `_at` suffix; foreign keys: `_id` suffix
- Pydantic schemas: `ResourceCreate`, `ResourceUpdate`, `ResourceRead`, `ResourceRequest`, `ResourceResult`
- Services: `<resource>_service.py`; Engines: `<purpose>_engine.py`
- API routes: versioned under `/api/v1`, plural kebab-case nouns (`/rank-fields`, `/soil-tests`)
- Booleans read as statements: `irrigation_available`, `is_active`, `has_errors`

## Priority Order

When speed, convenience, and quality conflict: **Correctness → Maintainability → Consistency → Speed**.
