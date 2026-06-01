# Hemp Adaptive Prescription Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static "is this field suitable?" XGBoost model with an adaptive prescription engine that learns from each field's completed crop cycles and handles optional soil analysis via imputation.

**Architecture:** A two-path engine: the cold-start path (XGBoost classifier + regressor, triggered when a field has no history or no lab analysis) produces an initial prescription, then the historical correction path adjusts yield and N/P/K recommendations using an exponential-weighted average of actual-vs-predicted outcomes from past cycles. Units throughout are dekar-based (Turkish standard: 1 ha = 10 dekar). The learning trigger is new lab analysis entry.

**Tech Stack:** Python 3.11, FastAPI, XGBoost 2.1.4, Pydantic v2, pytest

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `app/schemas/hemp_prescription.py` | Add variety/irrigation_type/previous_crop enums, dekar units, optional soil chemistry, confidence field |
| Create | `app/schemas/hemp_cycle.py` | `CycleRecord` (single completed cycle) + `AdaptivePrescriptionRequest` (request wrapper with history) |
| Modify | `app/ai/providers/ml/hemp_prescription.py` | Updated feature set (31 features), dekar constants, imputation for missing soil chemistry |
| Create | `app/ai/providers/ml/hemp_historical.py` | Exponential-weighted correction layer on top of cold-start result |
| Modify | `app/engines/hemp_prescription_engine.py` | Route cold-start vs historical; blockers handle optional fields; dekar notes |
| Modify | `app/api/hemp.py` | Add `POST /hemp/prescription/adaptive` endpoint |
| Modify | `scripts/generate_hemp_dataset.py` | New features + dekar units |
| Modify | `scripts/train_hemp_model.py` | Updated NUMERIC_FEATURES + CATEGORICAL_FEATURES |
| Modify | `REPORT/generate_report.py` | Full narrative rewrite: Kitt context, adaptive architecture, updated figures |
| Create | `tests/test_hemp_prescription.py` | All hemp prescription tests |

---

## Task 1 — Schema Layer

**Files:**
- Modify: `app/schemas/hemp_prescription.py`
- Create: `app/schemas/hemp_cycle.py`
- Create: `tests/test_hemp_prescription.py`

- [ ] **Step 1: Write failing schema tests**

```python
# tests/test_hemp_prescription.py
import pytest
from pydantic import ValidationError
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult
from app.schemas.hemp_cycle import CycleRecord, AdaptivePrescriptionRequest


def test_request_requires_area_dekar():
    with pytest.raises(ValidationError):
        HempPrescriptionRequest(slope_percent=5.0)  # missing area_dekar


def test_request_soil_chemistry_optional():
    # Should not raise -- soil chemistry fields are all optional
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)
    assert req.ph is None
    assert req.nitrogen_ppm is None


def test_request_defaults():
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)
    assert req.variety == "other"
    assert req.irrigation_type == "none"
    assert req.previous_crop == "other"
    assert req.first_hemp_season is False


def test_request_valid_variety():
    req = HempPrescriptionRequest(area_dekar=30.0, slope_percent=3.0, variety="narli")
    assert req.variety == "narli"


def test_request_invalid_variety():
    with pytest.raises(ValidationError):
        HempPrescriptionRequest(area_dekar=30.0, slope_percent=3.0, variety="unknown_variety")


def test_request_valid_irrigation_type():
    req = HempPrescriptionRequest(area_dekar=30.0, slope_percent=3.0, irrigation_type="drip")
    assert req.irrigation_type == "drip"


def test_result_has_confidence():
    result = HempPrescriptionResult(
        rec_nitrogen_kg_dekar=8.0, rec_phosphorus_kg_dekar=3.0,
        rec_potassium_kg_dekar=6.0, rec_irrigation_mm_week=5.0,
        expected_yield_ton_dekar=0.5, suitable=True, provider="xgboost",
    )
    assert result.confidence == 1.0  # default


def test_result_confidence_range():
    with pytest.raises(ValidationError):
        HempPrescriptionResult(
            rec_nitrogen_kg_dekar=8.0, rec_phosphorus_kg_dekar=3.0,
            rec_potassium_kg_dekar=6.0, rec_irrigation_mm_week=5.0,
            expected_yield_ton_dekar=0.5, suitable=True, provider="x",
            confidence=1.5,  # out of range
        )


def test_cycle_record_requires_all_fields():
    with pytest.raises(ValidationError):
        CycleRecord(cycle_id="c1")  # missing most fields


def test_cycle_record_valid():
    cycle = CycleRecord(
        cycle_id="c1",
        predicted_yield_ton_dekar=0.5, rec_nitrogen_kg_dekar=8.0,
        rec_phosphorus_kg_dekar=3.0, rec_potassium_kg_dekar=6.0,
        rec_irrigation_mm_week=5.0, applied_nitrogen_kg_dekar=7.5,
        applied_phosphorus_kg_dekar=3.0, applied_potassium_kg_dekar=6.0,
        applied_irrigation_mm_season=90.0, actual_yield_ton_dekar=0.52,
        actual_moisture_percent=12.0, cycle_days=120,
    )
    assert cycle.cycle_id == "c1"


def test_adaptive_request_no_history():
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)
    body = AdaptivePrescriptionRequest(request=req)
    assert body.cycle_history == []


def test_adaptive_request_with_history():
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)
    cycle = CycleRecord(
        cycle_id="c1", predicted_yield_ton_dekar=0.5, rec_nitrogen_kg_dekar=8.0,
        rec_phosphorus_kg_dekar=3.0, rec_potassium_kg_dekar=6.0,
        rec_irrigation_mm_week=5.0, applied_nitrogen_kg_dekar=7.5,
        applied_phosphorus_kg_dekar=3.0, applied_potassium_kg_dekar=6.0,
        applied_irrigation_mm_season=90.0, actual_yield_ton_dekar=0.55,
        actual_moisture_percent=12.0, cycle_days=120,
    )
    body = AdaptivePrescriptionRequest(request=req, cycle_history=[cycle])
    assert len(body.cycle_history) == 1
```

- [ ] **Step 2: Run tests to confirm they all fail**

```
pytest tests/test_hemp_prescription.py -v
```
Expected: ImportError or AttributeError -- schemas do not have these fields yet.

- [ ] **Step 3: Rewrite `app/schemas/hemp_prescription.py`**

```python
# app/schemas/hemp_prescription.py
"""Request and response schemas for the hemp prescription endpoint."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

DrainageClass  = Literal["poor", "moderate", "good", "excellent"]
TextureClass   = Literal["sandy loam", "loam", "silt loam", "clay loam", "silty clay loam"]
HempVariety    = Literal["narli", "vezir", "other"]
IrrigationType = Literal["drip", "sprinkler", "flood", "none"]
WaterSource    = Literal["well", "river", "municipal", "rain"]
PreviousCrop   = Literal["legume", "cereal", "sunflower", "hemp", "fallow", "other"]


class HempPrescriptionRequest(BaseModel):
    """All agronomic inputs required to generate a hemp prescription.

    Soil chemistry and climate fields are optional; the engine imputes
    variety-specific defaults and lowers the confidence score when they
    are absent.
    """
    model_config = ConfigDict(extra="forbid")

    # Variety and agronomic context
    variety: HempVariety = "other"
    first_hemp_season: bool = False
    previous_crop: PreviousCrop = "other"

    # Soil chemistry (optional -- allow no-lab cold start)
    ph:                      float | None = Field(None, ge=3.0,   le=10.0)
    nitrogen_ppm:            float | None = Field(None, ge=0.0)
    phosphorus_ppm:          float | None = Field(None, ge=0.0)
    potassium_ppm:           float | None = Field(None, ge=0.0)
    organic_matter_percent:  float | None = Field(None, ge=0.0, le=100.0)
    ec:                      float | None = Field(None, ge=0.0, description="Electrical conductivity (dS/m)")

    # Soil physical
    drainage_class: DrainageClass = "moderate"
    texture_class:  TextureClass  = "loam"

    # Field (area in dekar; 1 ha = 10 dekar)
    area_dekar:        float = Field(..., gt=0.0, description="Field area (dekar)")
    slope_percent:     float = Field(..., ge=0.0)
    irrigation_type:   IrrigationType = "none"
    water_source:      WaterSource    = "rain"
    elevation_meters:  float = Field(default=100.0, ge=0.0)

    # Climate (optional -- fetched from integrations when absent)
    avg_temp:              float | None = Field(None, description="Mean growing-season temperature (degC)")
    seasonal_rainfall_mm:  float | None = Field(None, ge=0.0)
    avg_humidity:          float | None = Field(None, ge=0.0, le=100.0)
    avg_solar_radiation:   float | None = Field(None, ge=0.0, description="MJ/m2/day")


class HempPrescriptionResult(BaseModel):
    """Prescription output returned for a hemp field scenario."""
    model_config = ConfigDict(extra="forbid")

    # Recommendations (dekar units)
    rec_nitrogen_kg_dekar:    float = Field(..., description="Recommended N application (kg/dekar)")
    rec_phosphorus_kg_dekar:  float = Field(..., description="Recommended P application (kg/dekar)")
    rec_potassium_kg_dekar:   float = Field(..., description="Recommended K application (kg/dekar)")
    rec_irrigation_mm_week:   float = Field(..., description="Weekly gross irrigation target (mm/week)")
    expected_yield_ton_dekar: float = Field(..., description="Forecast fiber yield (ton/dekar)")

    suitable:   bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0,
                              description="Prediction confidence: 1.0=full lab data+history, 0.4=no data")
    blockers:   list[str] = Field(default_factory=list)
    notes:      list[str] = Field(default_factory=list)
    provider:   str       = Field(..., description="Inference backend: xgboost | rule_based | historical | none")
```

- [ ] **Step 4: Create `app/schemas/hemp_cycle.py`**

```python
# app/schemas/hemp_cycle.py
"""Schemas for hemp cycle history and the adaptive prescription request."""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.hemp_prescription import HempPrescriptionRequest


class CycleRecord(BaseModel):
    """One completed crop cycle -- the atomic unit for historical learning.

    Stores what was recommended, what was actually applied, and what
    yield was achieved. The engine uses a list of these to adjust the
    cold-start recommendation for the next cycle.
    """
    model_config = ConfigDict(extra="forbid")

    cycle_id: str = Field(..., description="Unique cycle ID (e.g. FieldCropCycle PK as string)")

    # What the cold-start model recommended at cycle start
    predicted_yield_ton_dekar: float = Field(..., ge=0.0)
    rec_nitrogen_kg_dekar:     float = Field(..., ge=0.0)
    rec_phosphorus_kg_dekar:   float = Field(..., ge=0.0)
    rec_potassium_kg_dekar:    float = Field(..., ge=0.0)
    rec_irrigation_mm_week:    float = Field(..., ge=0.0)

    # What was actually applied
    applied_nitrogen_kg_dekar:    float = Field(..., ge=0.0)
    applied_phosphorus_kg_dekar:  float = Field(..., ge=0.0)
    applied_potassium_kg_dekar:   float = Field(..., ge=0.0)
    applied_irrigation_mm_season: float = Field(..., ge=0.0,
                                                description="Total irrigation applied over the full season (mm)")

    # Outcomes recorded at harvest
    actual_yield_ton_dekar:  float = Field(..., ge=0.0)
    actual_moisture_percent: float = Field(..., ge=0.0, le=100.0)
    cycle_days:              int   = Field(..., gt=0)

    # Agronomic context (used to weight relevance across varieties/systems)
    variety:        str = "other"
    previous_crop:  str = "other"
    irrigation_type: str = "none"


class AdaptivePrescriptionRequest(BaseModel):
    """Wrapper that bundles the current field request with cycle history."""
    model_config = ConfigDict(extra="forbid")

    request:       HempPrescriptionRequest
    cycle_history: list[CycleRecord] = Field(default_factory=list)
```

- [ ] **Step 5: Run tests -- all should pass**

```
pytest tests/test_hemp_prescription.py -v
```
Expected: all 13 tests PASS.

- [ ] **Step 6: Commit**

```
git add app/schemas/hemp_prescription.py app/schemas/hemp_cycle.py tests/test_hemp_prescription.py
git commit -m "feat(hemp): dekar units, variety/irrigation/rotation fields, optional soil chemistry, CycleRecord schema"
```

---

## Task 2 — Updated Cold-Start Provider

**Files:**
- Modify: `app/ai/providers/ml/hemp_prescription.py`

(Tests are added to the existing `tests/test_hemp_prescription.py`.)

- [ ] **Step 1: Add cold-start provider tests to the test file**

Append the following to `tests/test_hemp_prescription.py`:

```python
# ── Cold-start provider tests ──────────────────────────────────────────────────
from app.ai.providers.ml.hemp_prescription import (
    HempPrescriptionProvider, _ph_penalty, _temp_penalty,
    ROTATION_N_CREDIT, IRRIGATION_EFFICIENCY, VARIETY_YIELD_FACTOR,
)


def _base_request(**kwargs) -> HempPrescriptionRequest:
    defaults = dict(
        area_dekar=50.0, slope_percent=5.0, ph=6.5, nitrogen_ppm=30.0,
        phosphorus_ppm=20.0, potassium_ppm=150.0, organic_matter_percent=2.5,
        ec=1.5, avg_temp=20.0, seasonal_rainfall_mm=380.0,
        avg_humidity=58.0, avg_solar_radiation=16.0,
    )
    defaults.update(kwargs)
    return HempPrescriptionRequest(**defaults)


def test_ph_penalty_optimal():
    assert _ph_penalty(6.5) == 1.0


def test_ph_penalty_below_optimal():
    penalty = _ph_penalty(5.0)
    assert 0.0 < penalty < 1.0


def test_ph_penalty_above_optimal():
    penalty = _ph_penalty(8.0)
    assert 0.0 < penalty < 1.0


def test_temp_penalty_optimal():
    assert _temp_penalty(20.0) == 1.0


def test_temp_penalty_too_cold():
    assert _temp_penalty(5.0) < 1.0


def test_rotation_n_credit_legume():
    assert ROTATION_N_CREDIT["legume"] > 0.0


def test_rotation_n_credit_hemp_negative():
    assert ROTATION_N_CREDIT["hemp"] < 0.0


def test_irrigation_efficiency_drip_highest():
    assert IRRIGATION_EFFICIENCY["drip"] > IRRIGATION_EFFICIENCY["sprinkler"]
    assert IRRIGATION_EFFICIENCY["sprinkler"] > IRRIGATION_EFFICIENCY["flood"]
    assert IRRIGATION_EFFICIENCY["none"] == 0.0


def test_provider_rule_based_no_irrigation():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = _base_request(irrigation_type="none")
    result = provider.predict(req)
    assert result.rec_irrigation_mm_week == 0.0


def test_provider_rule_based_drip_prescribes_irrigation():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = _base_request(irrigation_type="drip", seasonal_rainfall_mm=200.0)
    result = provider.predict(req)
    assert result.rec_irrigation_mm_week > 0.0


def test_provider_rule_based_imputes_missing_soil():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)  # no soil chemistry
    result = provider.predict(req)
    assert result.suitable is True
    assert result.confidence < 0.75  # lower confidence when imputed


def test_provider_rule_based_legume_rotation_reduces_n():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req_other = _base_request(previous_crop="other")
    req_legume = _base_request(previous_crop="legume")
    n_other = provider.predict(req_other).rec_nitrogen_kg_dekar
    n_legume = provider.predict(req_legume).rec_nitrogen_kg_dekar
    assert n_legume < n_other


def test_provider_narli_variety_yields_more_than_other():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req_narli = _base_request(variety="narli")
    req_other = _base_request(variety="other")
    y_narli = provider.predict(req_narli).expected_yield_ton_dekar
    y_other = provider.predict(req_other).expected_yield_ton_dekar
    assert y_narli > y_other


def test_n_prescription_in_dekar_range():
    # Full N prescription should be roughly 0-12 kg/dekar (not 0-120 kg/ha)
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = _base_request(nitrogen_ppm=10.0, organic_matter_percent=1.0)  # low N soil
    result = provider.predict(req)
    assert 0.0 <= result.rec_nitrogen_kg_dekar <= 12.0


def test_yield_in_dekar_range():
    provider = HempPrescriptionProvider(model_dir="nonexistent_path")
    req = _base_request()
    result = provider.predict(req)
    assert 0.0 <= result.expected_yield_ton_dekar <= 1.0  # base max ~0.65 t/dekar
```

- [ ] **Step 2: Run new tests -- expect failures**

```
pytest tests/test_hemp_prescription.py -k "cold_start or rotation or irrigation_efficiency or rule_based or n_prescription or yield_in_dekar or impute or narli or legume" -v
```
Expected: most fail (old constants still in kg/ha).

- [ ] **Step 3: Rewrite `app/ai/providers/ml/hemp_prescription.py`**

```python
# app/ai/providers/ml/hemp_prescription.py
"""Nested XGBoost hemp prescription provider -- cold-start path.

Stage 1: suitability classifier (hemp_classifier.json)
Stage 2: yield regressor (hemp_regressor.json) -- suitable fields only.

N/P/K and irrigation are deterministic agronomic formulas (dekar units).
Missing soil chemistry is imputed with variety-specific defaults; confidence
score is lowered accordingly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult

try:
    from xgboost import Booster, DMatrix
    _XGBOOST_AVAILABLE = True
except ImportError:
    Booster = None  # type: ignore[assignment,misc]
    DMatrix = None  # type: ignore[assignment,misc]
    _XGBOOST_AVAILABLE = False

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[4] / "artifacts" / "hemp_model"

NUMERIC_FEATURES = [
    "ph", "nitrogen_ppm", "phosphorus_ppm", "potassium_ppm",
    "organic_matter_percent", "ec", "area_dekar", "slope_percent",
    "first_hemp_season", "elevation_meters", "avg_temp",
    "seasonal_rainfall_mm", "avg_humidity", "avg_solar_radiation",
    "rotation_n_credit",
]
CATEGORICAL_FEATURES = ["drainage_class", "texture_class", "variety", "irrigation_type"]

# Agronomic constants -- dekar units (1 ha = 10 dekar)
_PPM_TO_KG_DEKAR = 0.39      # 1 ppm ≈ 0.39 kg/dekar  (30 cm depth, 1.3 g/cm³)
_HEMP_N_TARGET   = 12.0       # kg/dekar
_HEMP_P_TARGET   =  5.0       # kg/dekar
_HEMP_K_TARGET   = 10.0       # kg/dekar
_HEMP_SEASONAL_WATER = 450.0  # mm  (area-independent)

ROTATION_N_CREDIT: dict[str, float] = {
    "legume":    3.0,
    "cereal":    0.0,
    "sunflower": 0.5,
    "hemp":     -1.0,
    "fallow":    1.0,
    "other":     0.0,
}

IRRIGATION_EFFICIENCY: dict[str, float] = {
    "drip":      0.90,
    "sprinkler": 0.78,
    "flood":     0.55,
    "none":      0.0,
}

VARIETY_YIELD_FACTOR: dict[str, float] = {
    "narli": 1.05,
    "vezir": 1.00,
    "other": 0.95,
}

# Default values used when soil chemistry or climate data are absent
_SOIL_DEFAULTS: dict[str, float] = {
    "ph": 6.5, "nitrogen_ppm": 30.0, "phosphorus_ppm": 20.0,
    "potassium_ppm": 150.0, "organic_matter_percent": 2.5, "ec": 1.5,
}
_CLIMATE_DEFAULTS: dict[str, float] = {
    "avg_temp": 20.0, "seasonal_rainfall_mm": 380.0,
    "avg_humidity": 58.0, "avg_solar_radiation": 16.0,
}


class HempPrescriptionProvider:
    """Nested XGBoost cold-start inference provider."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        self._classifier: Any | None = None
        self._regressor: Any | None = None
        self._feature_names: list[str] | None = None
        self._category_levels: dict[str, list[str]] | None = None

    def predict(self, request: HempPrescriptionRequest) -> HempPrescriptionResult:
        if _XGBOOST_AVAILABLE and self._model_dir.exists():
            return self._predict_xgboost(request)
        return self._predict_rule_based(request)

    # ── XGBoost path ─────────────────────────────────────────────────────────

    def _load_models(self) -> bool:
        meta = self._model_dir / "hemp_model_metadata.json"
        clf_path = self._model_dir / "hemp_classifier.json"
        reg_path = self._model_dir / "hemp_regressor.json"
        if not all(p.exists() for p in (meta, clf_path, reg_path)):
            return False
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            self._feature_names = data["feature_names"]
            self._category_levels = data["category_levels"]
            clf = Booster(); clf.load_model(str(clf_path)); self._classifier = clf
            reg = Booster(); reg.load_model(str(reg_path)); self._regressor = reg
            return True
        except Exception:
            self._classifier = self._regressor = None
            return False

    def _impute(self, request: HempPrescriptionRequest) -> tuple[dict, bool]:
        """Fill None fields with defaults. Returns (resolved_dict, had_any_missing)."""
        resolved: dict = {}
        had_missing = False
        for key, default in {**_SOIL_DEFAULTS, **_CLIMATE_DEFAULTS}.items():
            raw = getattr(request, key, None)
            if raw is None:
                resolved[key] = default
                had_missing = True
            else:
                resolved[key] = float(raw)
        return resolved, had_missing

    def _encode(self, request: HempPrescriptionRequest, imputed: dict) -> list[float]:
        assert self._category_levels is not None
        n_credit = ROTATION_N_CREDIT.get(request.previous_crop, 0.0)
        vector: list[float] = []
        for feat in NUMERIC_FEATURES:
            if feat == "rotation_n_credit":
                vector.append(float(n_credit))
            elif feat == "first_hemp_season":
                vector.append(1.0 if request.first_hemp_season else 0.0)
            elif feat == "area_dekar":
                vector.append(float(request.area_dekar))
            elif feat == "slope_percent":
                vector.append(float(request.slope_percent))
            elif feat == "elevation_meters":
                vector.append(float(request.elevation_meters))
            else:
                vector.append(float(imputed[feat]))
        for feat in CATEGORICAL_FEATURES:
            value = str(getattr(request, feat))
            for level in self._category_levels[feat]:
                vector.append(1.0 if value == level else 0.0)
        return vector

    def _predict_xgboost(self, request: HempPrescriptionRequest) -> HempPrescriptionResult:
        if self._classifier is None:
            if not self._load_models():
                return self._predict_rule_based(request)
        imputed, had_missing = self._impute(request)
        dmatrix = DMatrix([self._encode(request, imputed)], feature_names=self._feature_names)
        suitable_prob = float(self._classifier.predict(dmatrix)[0])
        suitable = suitable_prob >= 0.5
        confidence = 0.55 if had_missing else 0.80
        if not suitable:
            return HempPrescriptionResult(
                rec_nitrogen_kg_dekar=0.0, rec_phosphorus_kg_dekar=0.0,
                rec_potassium_kg_dekar=0.0, rec_irrigation_mm_week=0.0,
                expected_yield_ton_dekar=0.0, suitable=False,
                confidence=confidence, blockers=[], notes=[], provider="xgboost",
            )
        yield_raw = float(self._regressor.predict(dmatrix)[0])
        variety_factor = VARIETY_YIELD_FACTOR.get(request.variety, 1.0)
        expected_yield = round(max(0.0, yield_raw * variety_factor), 3)
        rec_n, rec_p, rec_k, rec_irr = self._compute_prescriptions(request, imputed)
        return HempPrescriptionResult(
            rec_nitrogen_kg_dekar=rec_n, rec_phosphorus_kg_dekar=rec_p,
            rec_potassium_kg_dekar=rec_k, rec_irrigation_mm_week=rec_irr,
            expected_yield_ton_dekar=expected_yield, suitable=True,
            confidence=confidence, blockers=[], notes=[], provider="xgboost",
        )

    # ── Rule-based fallback ────────────────────────────────────────────────────

    def _predict_rule_based(self, request: HempPrescriptionRequest) -> HempPrescriptionResult:
        imputed, had_missing = self._impute(request)
        rec_n, rec_p, rec_k, rec_irr = self._compute_prescriptions(request, imputed)
        yield_est = self._rule_based_yield(request, imputed)
        confidence = 0.40 if had_missing else 0.65
        return HempPrescriptionResult(
            rec_nitrogen_kg_dekar=rec_n, rec_phosphorus_kg_dekar=rec_p,
            rec_potassium_kg_dekar=rec_k, rec_irrigation_mm_week=rec_irr,
            expected_yield_ton_dekar=yield_est, suitable=True,
            confidence=confidence, blockers=[], notes=[], provider="rule_based",
        )

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _compute_prescriptions(
        self, request: HempPrescriptionRequest, imputed: dict
    ) -> tuple[float, float, float, float]:
        n_credit = ROTATION_N_CREDIT.get(request.previous_crop, 0.0)
        avail_n  = imputed["nitrogen_ppm"]           * _PPM_TO_KG_DEKAR
        avail_p  = imputed["phosphorus_ppm"]         * _PPM_TO_KG_DEKAR
        avail_k  = imputed["potassium_ppm"]          * _PPM_TO_KG_DEKAR
        n_from_om = imputed["organic_matter_percent"] * 2.0  # ~2 kg N per 1% OM per dekar

        rec_n = round(max(0.0, _HEMP_N_TARGET - avail_n - n_from_om - n_credit), 2)
        rec_p = round(max(0.0, _HEMP_P_TARGET - avail_p), 2)
        rec_k = round(max(0.0, _HEMP_K_TARGET - avail_k), 2)

        efficiency = IRRIGATION_EFFICIENCY.get(request.irrigation_type, 0.0)
        if efficiency == 0.0:
            rec_irr = 0.0
        else:
            effective_rainfall = imputed["seasonal_rainfall_mm"] * 0.75
            water_deficit = max(0.0, _HEMP_SEASONAL_WATER - effective_rainfall)
            rec_irr = round((water_deficit / 20.0) / efficiency, 2)

        return rec_n, rec_p, rec_k, rec_irr

    def _rule_based_yield(self, request: HempPrescriptionRequest, imputed: dict) -> float:
        ph_pen    = _ph_penalty(imputed["ph"])
        temp_pen  = _temp_penalty(imputed["avg_temp"])
        drain_f   = {"poor": 0.55, "moderate": 0.80, "good": 1.0, "excellent": 1.05}[request.drainage_class]
        slope_f   = max(0.6, 1.0 - max(0.0, request.slope_percent - 8.0) * 0.04)
        variety_f = VARIETY_YIELD_FACTOR.get(request.variety, 1.0)
        return round(0.65 * ph_pen * temp_pen * drain_f * slope_f * variety_f, 3)


def _ph_penalty(ph: float) -> float:
    if 6.0 <= ph <= 7.0:    return 1.0
    if ph < 6.0:             return max(0.0, 1.0 - (6.0 - ph) * 0.35)
    return                          max(0.0, 1.0 - (ph - 7.0) * 0.30)


def _temp_penalty(avg_temp: float) -> float:
    if 15.0 <= avg_temp <= 27.0: return 1.0
    if avg_temp < 15.0:           return max(0.0, 1.0 - (15.0 - avg_temp) * 0.06)
    return                               max(0.0, 1.0 - (avg_temp - 27.0) * 0.08)
```

- [ ] **Step 4: Run all tests**

```
pytest tests/test_hemp_prescription.py -v
```
Expected: all tests pass (XGBoost model not yet retrained, so rule-based path runs).

- [ ] **Step 5: Commit**

```
git add app/ai/providers/ml/hemp_prescription.py tests/test_hemp_prescription.py
git commit -m "feat(hemp): dekar units, optional soil chemistry imputation, variety/irrigation/rotation in cold-start provider"
```

---

## Task 3 — Historical Correction Provider

**Files:**
- Create: `app/ai/providers/ml/hemp_historical.py`

Tests appended to `tests/test_hemp_prescription.py`.

- [ ] **Step 1: Add historical provider tests**

Append to `tests/test_hemp_prescription.py`:

```python
# ── Historical provider tests ─────────────────────────────────────────────────
from app.ai.providers.ml.hemp_historical import HistoricalHempProvider
from app.schemas.hemp_cycle import CycleRecord


def _make_cycle(
    cycle_id: str,
    predicted: float,
    actual: float,
    rec_n: float = 8.0,
    applied_n: float = 8.0,
) -> CycleRecord:
    return CycleRecord(
        cycle_id=cycle_id,
        predicted_yield_ton_dekar=predicted,
        rec_nitrogen_kg_dekar=rec_n,
        rec_phosphorus_kg_dekar=3.0,
        rec_potassium_kg_dekar=6.0,
        rec_irrigation_mm_week=5.0,
        applied_nitrogen_kg_dekar=applied_n,
        applied_phosphorus_kg_dekar=3.0,
        applied_potassium_kg_dekar=6.0,
        applied_irrigation_mm_season=100.0,
        actual_yield_ton_dekar=actual,
        actual_moisture_percent=12.0,
        cycle_days=120,
    )


def _make_cold_start(yield_val: float = 0.50) -> HempPrescriptionResult:
    return HempPrescriptionResult(
        rec_nitrogen_kg_dekar=8.0, rec_phosphorus_kg_dekar=3.0,
        rec_potassium_kg_dekar=6.0, rec_irrigation_mm_week=5.0,
        expected_yield_ton_dekar=yield_val, suitable=True,
        confidence=0.80, provider="xgboost",
    )


def test_historical_no_cycles_returns_cold_start():
    provider = HistoricalHempProvider()
    req = _base_request()
    cold = _make_cold_start()
    result = provider.predict(req, [], cold)
    assert result is cold


def test_historical_upward_correction_when_actual_exceeds_predicted():
    provider = HistoricalHempProvider()
    req = _base_request()
    # actual 0.60 when predicted 0.50 -> factor = 1.2 -> corrected yield > cold start
    cycles = [_make_cycle("c1", predicted=0.50, actual=0.60)]
    cold = _make_cold_start(0.50)
    result = provider.predict(req, cycles, cold)
    assert result.expected_yield_ton_dekar > cold.expected_yield_ton_dekar


def test_historical_downward_correction_when_actual_below_predicted():
    provider = HistoricalHempProvider()
    req = _base_request()
    cycles = [_make_cycle("c1", predicted=0.50, actual=0.35)]
    cold = _make_cold_start(0.50)
    result = provider.predict(req, cycles, cold)
    assert result.expected_yield_ton_dekar < cold.expected_yield_ton_dekar


def test_historical_n_reduced_when_less_applied_with_good_yield():
    provider = HistoricalHempProvider()
    req = _base_request()
    # Applied 6.5 kg/dekar (rec=8.0) but actual/predicted ratio >= 0.93
    cycles = [_make_cycle("c1", predicted=0.50, actual=0.48, rec_n=8.0, applied_n=6.5)]
    cold = _make_cold_start(0.50)
    result = provider.predict(req, cycles, cold)
    assert result.rec_nitrogen_kg_dekar < cold.rec_nitrogen_kg_dekar


def test_historical_confidence_grows_with_more_cycles():
    provider = HistoricalHempProvider()
    req = _base_request()
    cold = _make_cold_start()
    one = provider.predict(req, [_make_cycle("c1", 0.5, 0.5)], cold)
    two = provider.predict(req, [_make_cycle("c1", 0.5, 0.5), _make_cycle("c2", 0.5, 0.5)], cold)
    assert two.confidence > one.confidence


def test_historical_confidence_capped_at_095():
    provider = HistoricalHempProvider()
    req = _base_request()
    cold = _make_cold_start()
    many_cycles = [_make_cycle(f"c{i}", 0.5, 0.5) for i in range(10)]
    result = provider.predict(req, many_cycles, cold)
    assert result.confidence <= 0.95


def test_historical_provider_field():
    provider = HistoricalHempProvider()
    req = _base_request()
    cycles = [_make_cycle("c1", 0.5, 0.5)]
    cold = _make_cold_start()
    result = provider.predict(req, cycles, cold)
    assert result.provider == "historical"


def test_historical_correction_factor_clamped():
    provider = HistoricalHempProvider()
    req = _base_request()
    # Extreme over-performance: actual = 5x predicted
    cycles = [_make_cycle("c1", predicted=0.10, actual=0.60)]
    cold = _make_cold_start(0.50)
    result = provider.predict(req, cycles, cold)
    # Factor clamped at 2.0 -> result <= 2 * cold_start
    assert result.expected_yield_ton_dekar <= cold.expected_yield_ton_dekar * 2.0 + 0.001
```

- [ ] **Step 2: Run new tests to confirm they all fail**

```
pytest tests/test_hemp_prescription.py -k "historical" -v
```
Expected: ImportError -- module does not exist yet.

- [ ] **Step 3: Create `app/ai/providers/ml/hemp_historical.py`**

```python
# app/ai/providers/ml/hemp_historical.py
"""Historical correction provider for hemp prescription.

Adjusts the cold-start XGBoost prediction using completed crop cycle data.
The core mechanism is an exponential-weighted average of (actual / predicted)
yield ratios across all cycles for the field, with more recent cycles
receiving higher weight.

The N correction uses the most recent cycle's efficiency signal: if the
farmer applied less N than recommended but still achieved near-expected
yield, the recommendation is reduced accordingly.
"""
from __future__ import annotations

from app.schemas.hemp_cycle import CycleRecord
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult

_DECAY = 0.6          # weight decay per cycle (older cycles count less)
_MIN_FACTOR = 0.50    # correction floor -- never recommend less than 50% of cold-start
_MAX_FACTOR = 2.00    # correction ceiling -- never recommend more than 2x cold-start
_BASE_CONFIDENCE = 0.65
_CONFIDENCE_PER_CYCLE = 0.10
_MAX_CONFIDENCE = 0.95


class HistoricalHempProvider:
    """Corrects cold-start predictions using actual field cycle outcomes.

    Pass ``cycles`` sorted oldest-first. The provider weights recent
    cycles more heavily via exponential decay.
    """

    def predict(
        self,
        request: HempPrescriptionRequest,
        cycles: list[CycleRecord],
        cold_start: HempPrescriptionResult,
    ) -> HempPrescriptionResult:
        if not cycles:
            return cold_start

        yield_factor = self._yield_correction_factor(cycles)
        n_factor     = self._n_correction_factor(cycles)

        corrected_yield = round(
            max(0.0, cold_start.expected_yield_ton_dekar * yield_factor), 3
        )
        corrected_n = round(
            max(0.0, cold_start.rec_nitrogen_kg_dekar * n_factor), 2
        )
        confidence = min(_MAX_CONFIDENCE, _BASE_CONFIDENCE + len(cycles) * _CONFIDENCE_PER_CYCLE)

        notes = list(cold_start.notes)
        notes.append(
            f"Recommendation adjusted from {len(cycles)} completed cycle(s) on this field. "
            f"Yield correction factor: {yield_factor:.2f}x."
        )

        return HempPrescriptionResult(
            rec_nitrogen_kg_dekar=corrected_n,
            rec_phosphorus_kg_dekar=cold_start.rec_phosphorus_kg_dekar,
            rec_potassium_kg_dekar=cold_start.rec_potassium_kg_dekar,
            rec_irrigation_mm_week=cold_start.rec_irrigation_mm_week,
            expected_yield_ton_dekar=corrected_yield,
            suitable=True,
            confidence=confidence,
            blockers=[],
            notes=notes,
            provider="historical",
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _yield_correction_factor(self, cycles: list[CycleRecord]) -> float:
        """Exponential-weighted average of (actual / predicted) ratios."""
        n = len(cycles)
        # cycles[0] is oldest; cycles[-1] is most recent
        weights = [_DECAY ** (n - 1 - i) for i in range(n)]
        total_weight = sum(weights)

        weighted_sum = 0.0
        usable_weight = 0.0
        for cycle, w in zip(cycles, weights):
            if cycle.predicted_yield_ton_dekar <= 0.0:
                continue
            ratio = cycle.actual_yield_ton_dekar / cycle.predicted_yield_ton_dekar
            weighted_sum  += ratio * w
            usable_weight += w

        if usable_weight == 0.0:
            return 1.0

        raw_factor = weighted_sum / usable_weight
        return max(_MIN_FACTOR, min(_MAX_FACTOR, raw_factor))

    def _n_correction_factor(self, cycles: list[CycleRecord]) -> float:
        """Adjust N recommendation based on the most recent cycle's efficiency signal.

        If the farmer applied meaningfully less N than recommended and still
        achieved at least 93% of the predicted yield, reduce the recommendation.
        If they applied more and got a clearly better yield, increase slightly.
        """
        recent = cycles[-1]
        rec = recent.rec_nitrogen_kg_dekar
        applied = recent.applied_nitrogen_kg_dekar

        if rec <= 0.0 or applied <= 0.0:
            return 1.0

        application_ratio = applied / rec
        yield_ratio = (
            recent.actual_yield_ton_dekar / max(0.001, recent.predicted_yield_ton_dekar)
        )

        # Applied <88% of recommendation yet yield held up >= 93% -> reduce recommendation
        if application_ratio < 0.88 and yield_ratio >= 0.93:
            return max(0.70, application_ratio + 0.05)

        # Applied >115% of recommendation AND yield improved >10% -> slight increase
        if application_ratio > 1.15 and yield_ratio > 1.10:
            return min(1.35, application_ratio * 0.90)

        return 1.0
```

- [ ] **Step 4: Run historical tests**

```
pytest tests/test_hemp_prescription.py -k "historical" -v
```
Expected: all historical tests PASS.

- [ ] **Step 5: Run full test file**

```
pytest tests/test_hemp_prescription.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```
git add app/ai/providers/ml/hemp_historical.py tests/test_hemp_prescription.py
git commit -m "feat(hemp): historical correction provider with exponential-weighted yield adjustment"
```

---

## Task 4 — Engine and API Update

**Files:**
- Modify: `app/engines/hemp_prescription_engine.py`
- Modify: `app/api/hemp.py`

Tests appended to `tests/test_hemp_prescription.py`.

- [ ] **Step 1: Add engine routing tests**

Append to `tests/test_hemp_prescription.py`:

```python
# ── Engine routing tests ──────────────────────────────────────────────────────
from app.engines.hemp_prescription_engine import compute_hemp_prescription
from app.schemas.hemp_cycle import CycleRecord


def test_engine_returns_result_no_history():
    req = _base_request()
    result = compute_hemp_prescription(req)
    assert result.suitable is True
    assert result.provider in ("xgboost", "rule_based")


def test_engine_blocker_ph_too_low():
    req = _base_request(ph=4.5)
    result = compute_hemp_prescription(req)
    assert result.suitable is False
    assert any("pH" in b or "ph" in b.lower() for b in result.blockers)


def test_engine_blocker_slope_excessive():
    req = _base_request(slope_percent=25.0)
    result = compute_hemp_prescription(req)
    assert result.suitable is False


def test_engine_no_blocker_when_ph_missing():
    # pH is None (not provided) -- should NOT trigger pH blocker
    req = HempPrescriptionRequest(area_dekar=50.0, slope_percent=5.0)
    result = compute_hemp_prescription(req)
    assert result.suitable is True  # no blocker for missing data


def test_engine_uses_historical_provider_when_cycles_provided():
    req = _base_request()
    cycle = _make_cycle("c1", predicted=0.50, actual=0.60)
    result = compute_hemp_prescription(req, cycle_history=[cycle])
    assert result.provider == "historical"


def test_engine_cold_start_when_no_cycles():
    req = _base_request()
    result = compute_hemp_prescription(req, cycle_history=[])
    assert result.provider in ("xgboost", "rule_based")


def test_engine_notes_attached():
    req = _base_request(ph=7.5)
    result = compute_hemp_prescription(req)
    assert isinstance(result.notes, list)
```

- [ ] **Step 2: Run new tests to confirm failures**

```
pytest tests/test_hemp_prescription.py -k "engine" -v
```
Expected: failures -- engine still uses old schema fields.

- [ ] **Step 3: Rewrite `app/engines/hemp_prescription_engine.py`**

```python
# app/engines/hemp_prescription_engine.py
"""Hemp prescription engine.

Validates inputs, fires blocking constraints, routes to cold-start or
historical provider, and attaches agronomic notes to the result.
"""
from __future__ import annotations

from pathlib import Path

from app.ai.providers.ml.hemp_historical import HistoricalHempProvider
from app.ai.providers.ml.hemp_prescription import HempPrescriptionProvider
from app.schemas.hemp_cycle import CycleRecord
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult

_cold_start_provider: HempPrescriptionProvider | None = None
_historical_provider: HistoricalHempProvider | None = None


def _get_cold_start(model_dir: str | Path | None = None) -> HempPrescriptionProvider:
    global _cold_start_provider
    if _cold_start_provider is None:
        _cold_start_provider = HempPrescriptionProvider(model_dir=model_dir)
    return _cold_start_provider


def _get_historical() -> HistoricalHempProvider:
    global _historical_provider
    if _historical_provider is None:
        _historical_provider = HistoricalHempProvider()
    return _historical_provider


# ── Hard blockers (deterministic -- run before any ML) ─────────────────────────

_BLOCKERS: list[tuple] = [
    (lambda r: r.ph is not None and r.ph < 5.0,
     "ph_too_acidic",    "Soil pH below 5.0 is outside hemp's tolerable range."),
    (lambda r: r.ph is not None and r.ph > 8.0,
     "ph_too_alkaline",  "Soil pH above 8.0 is outside hemp's tolerable range."),
    (lambda r: r.slope_percent > 20.0,
     "slope_excessive",  "Slope above 20% makes mechanical hemp cultivation impractical."),
    (lambda r: r.avg_temp is not None and r.avg_temp < 5.0,
     "temp_too_cold",    "Mean growing-season temperature below 5 degC prevents hemp establishment."),
    (lambda r: r.avg_temp is not None and r.avg_temp > 35.0,
     "temp_too_hot",     "Mean growing-season temperature above 35 degC causes heat stress."),
    (lambda r: r.ec is not None and r.ec > 4.0,
     "salinity_high",    "EC above 4.0 dS/m exceeds hemp's salinity tolerance."),
]


def _check_blockers(request: HempPrescriptionRequest) -> list[str]:
    return [message for check, _code, message in _BLOCKERS if check(request)]


# ── Agronomic notes ────────────────────────────────────────────────────────────

def _build_notes(request: HempPrescriptionRequest, result: HempPrescriptionResult) -> list[str]:
    notes: list[str] = []
    if request.ph is not None and not (6.0 <= request.ph <= 7.0):
        notes.append(f"pH {request.ph} is outside the ideal 6.0-7.0 range -- consider lime or sulfur amendment.")
    if request.drainage_class == "poor":
        notes.append("Poor drainage raises waterlogging risk -- raised beds or tile drainage recommended.")
    if result.rec_nitrogen_kg_dekar > 8.0:
        notes.append(f"High N deficit ({result.rec_nitrogen_kg_dekar:.1f} kg/dekar) -- split application across two dressings.")
    if request.irrigation_type == "none" and (request.seasonal_rainfall_mm or 999) < 300:
        notes.append("Seasonal rainfall below 300 mm with no irrigation -- drought stress likely during vegetative stage.")
    if request.slope_percent > 12:
        notes.append(f"Slope {request.slope_percent}% increases erosion risk -- contour cultivation advised.")
    if result.expected_yield_ton_dekar < 0.30:
        notes.append("Forecast yield below 0.30 t/dekar -- field conditions are suboptimal for fiber hemp.")
    if request.ph is None:
        notes.append("No soil analysis provided -- prescription based on variety defaults. Lab analysis recommended before planting.")
    return notes


# ── Public interface ───────────────────────────────────────────────────────────

def compute_hemp_prescription(
    request: HempPrescriptionRequest,
    *,
    model_dir: str | Path | None = None,
    cycle_history: list[CycleRecord] | None = None,
) -> HempPrescriptionResult:
    """Validate inputs, route to cold-start or historical provider, return annotated prescription."""
    blockers = _check_blockers(request)
    if blockers:
        return HempPrescriptionResult(
            rec_nitrogen_kg_dekar=0.0, rec_phosphorus_kg_dekar=0.0,
            rec_potassium_kg_dekar=0.0, rec_irrigation_mm_week=0.0,
            expected_yield_ton_dekar=0.0, suitable=False,
            confidence=1.0, blockers=blockers, notes=[], provider="none",
        )

    cold_start = _get_cold_start(model_dir).predict(request)

    if not cold_start.suitable:
        return HempPrescriptionResult(
            rec_nitrogen_kg_dekar=0.0, rec_phosphorus_kg_dekar=0.0,
            rec_potassium_kg_dekar=0.0, rec_irrigation_mm_week=0.0,
            expected_yield_ton_dekar=0.0, suitable=False,
            confidence=cold_start.confidence, blockers=[],
            notes=["Field conditions do not meet the minimum suitability threshold for hemp cultivation."],
            provider=cold_start.provider,
        )

    if cycle_history:
        result = _get_historical().predict(request, cycle_history, cold_start)
    else:
        result = cold_start

    agronomic_notes = _build_notes(request, result)
    return result.model_copy(update={"notes": agronomic_notes + result.notes})
```

- [ ] **Step 4: Update `app/api/hemp.py`**

```python
# app/api/hemp.py
"""Hemp prescription API endpoints."""
from fastapi import APIRouter, HTTPException

from app.engines.hemp_prescription_engine import compute_hemp_prescription
from app.schemas.hemp_cycle import AdaptivePrescriptionRequest
from app.schemas.hemp_prescription import HempPrescriptionRequest, HempPrescriptionResult

router = APIRouter(prefix="/hemp", tags=["hemp"])


@router.post("/prescription", response_model=HempPrescriptionResult)
def hemp_prescription_endpoint(request: HempPrescriptionRequest) -> HempPrescriptionResult:
    """Cold-start prescription from soil, field, and climate inputs only."""
    try:
        return compute_hemp_prescription(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/prescription/adaptive", response_model=HempPrescriptionResult)
def hemp_adaptive_prescription_endpoint(body: AdaptivePrescriptionRequest) -> HempPrescriptionResult:
    """Adaptive prescription: cold-start corrected by completed cycle history.

    Pass cycles sorted oldest-first in ``cycle_history``.
    When ``cycle_history`` is empty, behaves identically to POST /prescription.
    """
    try:
        return compute_hemp_prescription(
            body.request,
            cycle_history=body.cycle_history or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
```

- [ ] **Step 5: Run all tests**

```
pytest tests/test_hemp_prescription.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```
git add app/engines/hemp_prescription_engine.py app/api/hemp.py tests/test_hemp_prescription.py
git commit -m "feat(hemp): adaptive engine routing (cold-start vs historical), optional-field blockers, /prescription/adaptive endpoint"
```

---

## Task 5 — Dataset, Training, and Model Retrain

**Files:**
- Modify: `scripts/generate_hemp_dataset.py`
- Modify: `scripts/train_hemp_model.py`
- Run: generate + train

- [ ] **Step 1: Rewrite `scripts/generate_hemp_dataset.py`**

```python
"""
Hemp prescription synthetic dataset generator -- dekar units.

2,000 samples covering diverse soil, field, and climate conditions.
New in this version: variety (narli/vezir/other), irrigation_type, water_source,
previous_crop, first_hemp_season, rotation_n_credit. Units: kg/dekar, ton/dekar.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from random import Random

RANDOM_SEED = 42
N_SAMPLES   = 2000
OUTPUT_PATH = Path("data/hemp_training.csv")

# Dekar-unit constants (1 ha = 10 dekar)
HEMP_N_TARGET_KG_DEKAR  = 12.0
HEMP_P_TARGET_KG_DEKAR  =  5.0
HEMP_K_TARGET_KG_DEKAR  = 10.0
HEMP_SEASONAL_WATER_MM  = 450.0
PPM_TO_KG_DEKAR         =  0.39   # 30 cm depth, 1.3 g/cm³
SUITABILITY_THRESHOLD   =  0.30   # ton/dekar
BASE_YIELD              =  0.65   # ton/dekar

DRAINAGE_CLASSES  = ["poor", "moderate", "good", "excellent"]
TEXTURE_CLASSES   = ["sandy loam", "loam", "silt loam", "clay loam", "silty clay loam"]
VARIETIES         = ["narli", "vezir", "other"]
IRRIGATION_TYPES  = ["drip", "sprinkler", "flood", "none"]
WATER_SOURCES     = ["well", "river", "municipal", "rain"]
PREVIOUS_CROPS    = ["legume", "cereal", "sunflower", "hemp", "fallow", "other"]

ROTATION_N_CREDIT = {"legume": 3.0, "cereal": 0.0, "sunflower": 0.5, "hemp": -1.0, "fallow": 1.0, "other": 0.0}
IRRIGATION_EFF    = {"drip": 0.90, "sprinkler": 0.78, "flood": 0.55, "none": 0.0}
VARIETY_FACTOR    = {"narli": 1.05, "vezir": 1.00, "other": 0.95}


def _ph_penalty(ph: float) -> float:
    if 6.0 <= ph <= 7.0: return 1.0
    if ph < 6.0:          return max(0.0, 1.0 - (6.0 - ph) * 0.35)
    return                       max(0.0, 1.0 - (ph - 7.0) * 0.30)


def _temp_penalty(avg_temp: float) -> float:
    if 15.0 <= avg_temp <= 27.0: return 1.0
    if avg_temp < 15.0:           return max(0.0, 1.0 - (15.0 - avg_temp) * 0.06)
    return                               max(0.0, 1.0 - (avg_temp - 27.0) * 0.08)


def _drainage_factor(dc: str) -> float:
    return {"poor": 0.55, "moderate": 0.80, "good": 1.0, "excellent": 1.05}[dc]


def _slope_factor(slope: float) -> float:
    return max(0.6, 1.0 - max(0.0, slope - 8.0) * 0.04)


def generate_row(rng: Random, row_id: int) -> dict:
    # Soil
    ph                    = round(rng.uniform(5.0, 8.0), 2)
    nitrogen_ppm          = round(rng.uniform(15.0, 80.0), 1)
    phosphorus_ppm        = round(rng.uniform(8.0, 45.0), 1)
    potassium_ppm         = round(rng.uniform(60.0, 280.0), 1)
    organic_matter_percent= round(rng.uniform(1.0, 6.5), 2)
    ec                    = round(rng.uniform(0.1, 4.0), 2)
    drainage_class        = rng.choice(DRAINAGE_CLASSES)
    texture_class         = rng.choice(TEXTURE_CLASSES)

    # Field (area in dekar: 20-600 dekar = 2-60 ha)
    area_dekar            = round(rng.uniform(20.0, 600.0), 1)
    slope_percent         = round(rng.uniform(0.5, 22.0), 1)
    irrigation_type       = rng.choice(IRRIGATION_TYPES)
    water_source          = rng.choice(WATER_SOURCES)
    elevation_meters      = round(rng.uniform(20.0, 1100.0), 1)

    # Variety and context
    variety               = rng.choice(VARIETIES)
    previous_crop         = rng.choice(PREVIOUS_CROPS)
    first_hemp_season     = int(rng.random() < 0.4)  # 40% first-timers

    # Climate
    avg_temp              = round(rng.uniform(8.0, 35.0), 1)
    seasonal_rainfall_mm  = round(rng.uniform(150.0, 650.0), 1)
    avg_humidity          = round(rng.uniform(35.0, 85.0), 1)
    avg_solar_radiation   = round(rng.uniform(10.0, 24.0), 1)

    # Rotation N credit
    rotation_n_credit = ROTATION_N_CREDIT.get(previous_crop, 0.0)

    # Prescription (dekar units)
    avail_n   = nitrogen_ppm   * PPM_TO_KG_DEKAR
    avail_p   = phosphorus_ppm * PPM_TO_KG_DEKAR
    avail_k   = potassium_ppm  * PPM_TO_KG_DEKAR
    n_from_om = organic_matter_percent * 2.0

    rec_n = max(0.0, HEMP_N_TARGET_KG_DEKAR - avail_n - n_from_om - rotation_n_credit)
    rec_p = max(0.0, HEMP_P_TARGET_KG_DEKAR - avail_p)
    rec_k = max(0.0, HEMP_K_TARGET_KG_DEKAR - avail_k)

    eff = IRRIGATION_EFF.get(irrigation_type, 0.0)
    if eff == 0.0:
        irrigation_mm_week = 0.0
    else:
        deficit = max(0.0, HEMP_SEASONAL_WATER_MM - seasonal_rainfall_mm * 0.75)
        irrigation_mm_week = (deficit / 20.0) / eff

    # Yield (deterministic core, then Gaussian noise)
    variety_f = VARIETY_FACTOR.get(variety, 1.0)
    raw_yield = (
        BASE_YIELD
        * _ph_penalty(ph)
        * _temp_penalty(avg_temp)
        * _drainage_factor(drainage_class)
        * _slope_factor(slope_percent)
        * variety_f
    )

    suitable = 1 if raw_yield >= SUITABILITY_THRESHOLD else 0

    def noisy(value: float, sigma_pct: float) -> float:
        return round(max(0.0, value * (1.0 + rng.gauss(0.0, sigma_pct))), 3)

    return {
        "sample_id": row_id,
        "ph": ph, "nitrogen_ppm": nitrogen_ppm, "phosphorus_ppm": phosphorus_ppm,
        "potassium_ppm": potassium_ppm, "organic_matter_percent": organic_matter_percent,
        "ec": ec, "drainage_class": drainage_class, "texture_class": texture_class,
        "area_dekar": area_dekar, "slope_percent": slope_percent,
        "irrigation_type": irrigation_type, "water_source": water_source,
        "elevation_meters": elevation_meters, "variety": variety,
        "previous_crop": previous_crop, "first_hemp_season": first_hemp_season,
        "avg_temp": avg_temp, "seasonal_rainfall_mm": seasonal_rainfall_mm,
        "avg_humidity": avg_humidity, "avg_solar_radiation": avg_solar_radiation,
        "rotation_n_credit": rotation_n_credit,
        "rec_nitrogen_kg_dekar":   noisy(rec_n, 0.08),
        "rec_phosphorus_kg_dekar": noisy(rec_p, 0.08),
        "rec_potassium_kg_dekar":  noisy(rec_k, 0.08),
        "rec_irrigation_mm_week":  noisy(irrigation_mm_week, 0.10),
        "expected_yield_ton_dekar": noisy(raw_yield, 0.12),
        "suitable": suitable,
    }


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> list[dict]:
    rng = Random(seed)
    return [generate_row(rng, i) for i in range(n_samples)]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_dataset()
    fieldnames = list(rows[0].keys())
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    n_suitable = sum(r["suitable"] for r in rows)
    print(f"Generated {len(rows)} rows -> {OUTPUT_PATH}")
    print(f"  Suitable:   {n_suitable} ({100 * n_suitable / len(rows):.1f}%)")
    print(f"  Unsuitable: {len(rows) - n_suitable} ({100 * (len(rows)-n_suitable)/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rewrite `scripts/train_hemp_model.py` feature lists**

In `scripts/train_hemp_model.py`, replace only the constants block (lines 49-79) with:

```python
NUMERIC_FEATURES = [
    "ph", "nitrogen_ppm", "phosphorus_ppm", "potassium_ppm",
    "organic_matter_percent", "ec", "area_dekar", "slope_percent",
    "first_hemp_season", "elevation_meters", "avg_temp",
    "seasonal_rainfall_mm", "avg_humidity", "avg_solar_radiation",
    "rotation_n_credit",
]
CATEGORICAL_FEATURES = ["drainage_class", "texture_class", "variety", "irrigation_type"]

CLASSIFIER_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "alpha": 0.1,
    "lambda": 1.0,
}
CLASSIFIER_ROUNDS = 150

REGRESSOR_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "alpha": 0.05,
    "lambda": 1.0,
}
REGRESSOR_ROUNDS = 150

TEST_FRACTION = 0.20
```

Also update the target column references in `main()`:

Find `y_train_cls = [float(r["suitable"]) for r in train_rows]` -- this stays the same.

Find `y_train_reg = [float(r["expected_yield_ton_ha"]) for r, _ in suitable_train]` -- change to:
```python
y_train_reg = [float(r["expected_yield_ton_dekar"]) for r, _ in suitable_train]
y_test_reg  = [float(r["expected_yield_ton_dekar"]) for r, _ in suitable_test]
```

- [ ] **Step 3: Generate new dataset and retrain**

```
cd "C:\Users\user\Desktop\YoruzuyAI\Side Projects\AgriMind"
python scripts/generate_hemp_dataset.py
python scripts/train_hemp_model.py
```

Expected output (approximate):
```
Generated 2000 rows -> data\hemp_training.csv
  Suitable:   ~1150 (57.5%)
  Unsuitable: ~850  (42.5%)
...
[Stage 1] Training suitability classifier ...
  Accuracy:  ~0.93+
  F1 Score:  ~0.94+
[Stage 2] Training yield regressor ...
  RMSE: ~0.05 t/dekar
  R²:   ~0.79+
```

- [ ] **Step 4: Verify XGBoost path works end-to-end**

```python
# Quick smoke test -- run this in Python REPL or as a script
from app.schemas.hemp_prescription import HempPrescriptionRequest
from app.engines.hemp_prescription_engine import compute_hemp_prescription

req = HempPrescriptionRequest(
    area_dekar=50.0, slope_percent=5.0, ph=6.5, nitrogen_ppm=30.0,
    phosphorus_ppm=20.0, potassium_ppm=150.0, organic_matter_percent=2.5,
    ec=1.5, avg_temp=20.0, seasonal_rainfall_mm=380.0,
    avg_humidity=58.0, avg_solar_radiation=16.0,
    variety="narli", irrigation_type="drip", previous_crop="cereal",
)
result = compute_hemp_prescription(req)
print(result)
assert result.provider == "xgboost"
assert 0.0 < result.expected_yield_ton_dekar < 1.0
assert result.rec_nitrogen_kg_dekar <= 12.0
print("Smoke test passed.")
```

Run: `python -c "exec(open('smoke_test.py').read())"` (save above as `smoke_test.py` temporarily)

- [ ] **Step 5: Run full test suite**

```
pytest tests/test_hemp_prescription.py -v
```
Expected: all tests PASS (now using XGBoost path for tests with real soil values).

- [ ] **Step 6: Commit**

```
git add scripts/generate_hemp_dataset.py scripts/train_hemp_model.py data/hemp_training.csv artifacts/hemp_model/
git commit -m "feat(hemp): regenerate dataset and retrain with dekar units, variety, irrigation_type, rotation_n_credit (31 features)"
```

---

## Task 6 — Report Update

**Files:**
- Modify: `REPORT/generate_report.py`
- Run: `python REPORT/generate_report.py`

- [ ] **Step 1: Update narrative constants and text in `generate_report.py`**

The report needs these changes throughout (all are find-and-replace + section rewrites):

**Cover + Abstract:** Replace "AgriMind Hemp Module" framing with "AgriMind / Kitt Platform" context. Add that the system runs inside Kitt, a B2B hemp supply-chain platform. Mention dekar units explicitly.

**Section 1 (Introduction):** Add:
- Kitt platform context (2-3 sentences before hemp basics)
- Hemp variety table: Narlı (early maturing, higher yield), Vezir (fiber-focused), Other
- Irrigation types and their efficiency: drip 90%, sprinkler 78%, flood 55%, none
- Crop rotation nitrogen credit table

**Section 4 (Architecture):** Update `fig_architecture()` to add a third path for "Historical Correction" that sits between the cold-start regressor and the final output box.

**Section 5 (Dataset):** Update Table 2 with new features: variety, irrigation_type, water_source, previous_crop, first_hemp_season, rotation_n_credit. Change area to dekar, yield to ton/dekar.

**Section 7 (Prescription Formulas):** Update all formulas and constants to dekar units. Add irrigation efficiency formula. Add rotation N credit formula.

**New Section 7.5 (Historical Correction Layer):** Add between Sections 7 and 8:
- Explain CycleRecord structure
- Exponential-weighted average formula:

```
factor = sum_i( w_i * (actual_i / predicted_i) ) / sum_i(w_i)
where w_i = DECAY^(N-1-i),  DECAY = 0.6,  i=0 oldest ... N-1 newest
factor clamped to [0.50, 2.00]
```

- N correction rule (efficiency signal from most recent cycle)
- Confidence formula: min(0.95, 0.65 + n_cycles * 0.10)

**Section 8 (Results):** Update all metrics to use new model output (run evaluate script after training to get updated numbers). Add row for `confidence` column.

**Section 9 (Discussion):** Add paragraph on why historical correction is separate from the XGBoost model (interpretability, data volume per field is small, correction is explainable to agronomist).

- [ ] **Step 2: Run updated report generator**

```
python REPORT/generate_report.py
```
Expected: `Saved: ...REPORT/SCI305_Final_Report.docx` with no errors.

- [ ] **Step 3: Open and spot-check the DOCX**

Verify:
- Cover page has "Kitt Platform" reference
- Table 1 includes Narlı / Vezir variety rows
- Dekar units appear throughout (not hectares)
- Section 7.5 (Historical Correction) is present with the ewma formula
- All figures render without errors
- References section intact

- [ ] **Step 4: Commit**

```
git add REPORT/generate_report.py REPORT/SCI305_Final_Report.docx
git commit -m "docs(report): update to Kitt platform context, dekar units, adaptive architecture, historical correction section"
```

---

## Self-Review

### Spec coverage check

| Requirement | Covered by |
|---|---|
| Dekar units | Task 1 (schema), Task 2 (provider), Task 5 (dataset) |
| Hemp varieties (Narlı/Vezir/Diğer) | Task 1 (schema), Task 2 (VARIETY_YIELD_FACTOR) |
| Irrigation type (drip/sprinkler/flood/none) | Task 1 (schema), Task 2 (IRRIGATION_EFFICIENCY) |
| Water source | Task 1 (schema) -- stored but not ML feature in V1 |
| Previous crop / rotation | Task 1 (schema), Task 2 (ROTATION_N_CREDIT) |
| First hemp season flag | Task 1 (schema), Task 2 (numeric feature) |
| Optional soil chemistry | Task 1 (schema), Task 2 (imputation + confidence) |
| Historical cycle learning | Task 3 (HistoricalHempProvider) |
| Exponential-weighted correction | Task 3 (\_yield\_correction\_factor) |
| N efficiency signal | Task 3 (\_n\_correction\_factor) |
| Confidence score | Task 1 (result schema), Task 2+3 (assigned) |
| Engine routing cold-start vs historical | Task 4 |
| /prescription/adaptive endpoint | Task 4 |
| No blocker fired for missing pH | Task 4 (lambda guards) |
| Dataset + retrain | Task 5 |
| Report: Kitt context, dekar, historical section | Task 6 |

### Placeholder scan

No TBDs or TODOs in task steps. All code blocks are complete. All commands include expected output.

### Type consistency

- `HempPrescriptionRequest` uses `area_dekar` throughout Tasks 1-5. No stale `area_hectares` reference.
- `HempPrescriptionResult` uses `rec_nitrogen_kg_dekar` throughout. Confirmed in Tasks 1, 2, 3, 4.
- `CycleRecord` field names match usage in `HistoricalHempProvider._n_correction_factor` (Task 3).
- `compute_hemp_prescription` signature in Task 4 matches usage in Task 4 API + Task 4 tests.
- `NUMERIC_FEATURES` in `train_hemp_model.py` (Task 5) matches `hemp_prescription.py` provider (Task 2). Both use `area_dekar` and `rotation_n_credit`.
