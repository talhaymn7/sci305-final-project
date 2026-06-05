# AgriMind

AgriMind is an AI-powered agricultural decision support platform built with FastAPI. It ranks fields for crop suitability, generates agronomic recommendations, and predicts hemp (*Cannabis sativa* L.) yield using a nested XGBoost classification–regression pipeline.

---

## Overview

The system addresses a practical gap in precision agriculture: industrial hemp has been largely excluded from systematic yield research due to decades of legal restrictions. AgriMind applies machine learning methods that are well-established for staple crops (maize, wheat) to hemp, combining deterministic agronomic scoring with a two-stage ML pipeline.

**Two-stage hemp yield model:**
- **Stage 1 — Suitability Classifier:** XGBoost binary classifier that determines whether a field is viable for hemp cultivation (F1: 0.961, Accuracy: 0.952)
- **Stage 2 — Yield Regressor:** XGBoost regressor trained only on suitable fields; predicts expected yield in t/dekar (RMSE: 0.065, R²: 0.70)
- **Historical correction layer:** exponential-weighted adjustment using actual outcomes from prior growing seasons on the same field

---

## Architecture

```
app/
├── api/            # FastAPI routes (HTTP parsing, error handling)
├── services/       # Database reads/writes, domain logic
├── engines/        # Deterministic scoring (suitability, ranking, explanation)
├── ai/
│   ├── contracts/  # Abstract provider interfaces
│   ├── providers/  # rule_based · ml (XGBoost) · llm (OpenAI) · stub
│   ├── orchestration/  # End-to-end workflows
│   └── registry.py     # Provider selection via env vars
├── models/         # SQLAlchemy ORM models
├── schemas/        # Pydantic I/O contracts
└── ingestion/      # NASA POWER & FAOSTAT data pipeline
config/
└── scoring_weights.json   # Suitability dimension weights
scripts/
├── generate_hemp_dataset.py   # Synthetic training data (2,000 samples)
└── train_hemp_model.py        # Trains Stage 1 + Stage 2 XGBoost models
migrations/                    # Alembic migration history
tests/                         # pytest test suite
```

---

## Scoring Engine

Field suitability is scored across six weighted dimensions:

| Dimension              | Weight |
|------------------------|--------|
| Soil compatibility     | 24     |
| pH compatibility       | 20     |
| Climate compatibility  | 20     |
| Water availability     | 16     |
| Drainage compatibility | 12     |
| Slope compatibility    | 8      |

Minimum field area is a **hard constraint** — undersized fields are excluded regardless of other scores.

---

## External Data Sources

- **NASA POWER** — daily meteorological data (temperature, rainfall, humidity, solar radiation) fetched per field coordinates
- **FAOSTAT** — annual crop production and yield statistics for economic scoring

---

## Pluggable AI Providers

Provider selection is via `.env`:

| Variable              | Options                              |
|-----------------------|--------------------------------------|
| `YIELD_PROVIDER`      | `stub` · `ml` · `xgboost`           |
| `EXPLANATION_PROVIDER`| `deterministic` · `rule_based`      |
| `RISK_PROVIDER`       | `rule_based` · `stub`               |
| `EXTRACTION_PROVIDER` | `manual` · `rule_based` · `stub`    |
| `AI_ASSISTANT_PROVIDER`| `openai`                            |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Apply database migrations
alembic upgrade head

# Seed sample data
python seed.py

# Generate hemp training data and train models
python scripts/generate_hemp_dataset.py
python scripts/train_hemp_model.py

# Start the API server
uvicorn app.main:app --reload
# API available at http://localhost:8000/api/v1
```

---

## API Endpoints

| Method | Endpoint                                          | Description                              |
|--------|---------------------------------------------------|------------------------------------------|
| POST   | `/api/v1/rank-fields/`                            | Rank multiple fields for a crop          |
| GET    | `/api/v1/recommendation/{field_id}/{crop_id}`     | Single field/crop recommendation         |
| GET    | `/api/v1/fields/{field_id}/management-plan`       | Weekly irrigation & fertilizer plan      |
| POST   | `/api/v1/agri-assistant/ask`                      | LLM-powered agronomic Q&A                |
| POST   | `/api/v1/hemp/prescription`                       | Hemp-specific yield & prescription       |

---

## Tests

```bash
pytest                          # run all tests
pytest tests/test_ranking.py    # single file
```

Tests use an in-memory SQLite database. Fixtures are in `tests/conftest.py`.

---

## Key Dependencies

- **FastAPI** — web framework
- **SQLAlchemy + Alembic** — ORM and migrations
- **XGBoost** — nested yield prediction pipeline
- **Pydantic** — schema validation
- **httpx** — async HTTP client for external API calls
