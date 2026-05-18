"""XGBoost-backed hemp prescription provider.

Loads five independent regressors from artifacts/hemp_model/ and runs
inference for all targets in a single call. Falls back gracefully to
rule-based estimates when model artifacts are missing.
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

TARGETS = [
    "rec_nitrogen_kg_ha",
    "rec_phosphorus_kg_ha",
    "rec_potassium_kg_ha",
    "rec_irrigation_mm_week",
    "expected_yield_ton_ha",
]

NUMERIC_FEATURES = [
    "ph", "nitrogen_ppm", "phosphorus_ppm", "potassium_ppm",
    "organic_matter_percent", "ec", "area_hectares", "slope_percent",
    "irrigation_available", "elevation_meters", "avg_temp",
    "seasonal_rainfall_mm", "avg_humidity", "avg_solar_radiation",
]

CATEGORICAL_FEATURES = ["drainage_class", "texture_class"]


class HempPrescriptionProvider:
    """Inference provider for hemp prescription models."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self._model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        self._models: dict[str, Any] | None = None
        self._feature_names: list[str] | None = None
        self._category_levels: dict[str, list[str]] | None = None

    def predict(self, request: HempPrescriptionRequest) -> HempPrescriptionResult:
        if _XGBOOST_AVAILABLE and self._model_dir.exists():
            return self._predict_xgboost(request)
        return self._predict_rule_based(request)

    # ── XGBoost path ───────────────────────────────────────────────────────────

    def _load_models(self) -> bool:
        metadata_path = self._model_dir / "hemp_model_metadata.json"
        if not metadata_path.exists():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._feature_names = metadata["feature_names"]
            self._category_levels = metadata["category_levels"]
            self._models = {}
            for target in TARGETS:
                model_path = self._model_dir / f"{target}.json"
                if not model_path.exists():
                    return False
                booster = Booster()
                booster.load_model(str(model_path))
                self._models[target] = booster
            return True
        except Exception:
            self._models = None
            return False

    def _encode(self, request: HempPrescriptionRequest) -> list[float]:
        assert self._category_levels is not None
        vector: list[float] = []

        for feature in NUMERIC_FEATURES:
            value = getattr(request, feature)
            vector.append(float(int(value) if isinstance(value, bool) else value))

        for feature in CATEGORICAL_FEATURES:
            value = getattr(request, feature)
            for level in self._category_levels[feature]:
                vector.append(1.0 if value == level else 0.0)

        return vector

    def _predict_xgboost(self, request: HempPrescriptionRequest) -> HempPrescriptionResult:
        if self._models is None:
            if not self._load_models():
                return self._predict_rule_based(request)

        vector = self._encode(request)
        dmatrix = DMatrix([vector], feature_names=self._feature_names)

        predictions = {
            target: float(max(0.0, self._models[target].predict(dmatrix)[0]))
            for target in TARGETS
        }
        return HempPrescriptionResult(
            **predictions,
            suitable=True,
            blockers=[],
            notes=[],
            provider="xgboost",
        )

    # ── Rule-based fallback ────────────────────────────────────────────────────

    def _predict_rule_based(self, request: HempPrescriptionRequest) -> HempPrescriptionResult:
        """Deterministic prescription using the same agronomic formulas as the data generator."""
        PPM_TO_KG_HA = 3.9
        HEMP_N_TARGET = 120.0
        HEMP_P_TARGET = 50.0
        HEMP_K_TARGET = 100.0
        HEMP_SEASONAL_WATER = 450.0

        avail_n = request.nitrogen_ppm * PPM_TO_KG_HA
        avail_p = request.phosphorus_ppm * PPM_TO_KG_HA
        avail_k = request.potassium_ppm * PPM_TO_KG_HA
        n_from_om = request.organic_matter_percent * 20.0

        rec_n = max(0.0, HEMP_N_TARGET - avail_n - n_from_om)
        rec_p = max(0.0, HEMP_P_TARGET - avail_p)
        rec_k = max(0.0, HEMP_K_TARGET - avail_k)

        effective_rainfall = request.seasonal_rainfall_mm * 0.75
        water_deficit = max(0.0, HEMP_SEASONAL_WATER - effective_rainfall)
        rec_irrigation = (water_deficit / 20.0) if request.irrigation_available else 0.0

        ph_pen = _ph_penalty(request.ph)
        temp_pen = _temp_penalty(request.avg_temp)
        drain_f = {"poor": 0.55, "moderate": 0.80, "good": 1.0, "excellent": 1.05}[request.drainage_class]
        slope_f = max(0.6, 1.0 - max(0.0, request.slope_percent - 8.0) * 0.04)
        n_suff = min(1.0, (avail_n + n_from_om + rec_n) / HEMP_N_TARGET)
        p_suff = min(1.0, (avail_p + rec_p) / HEMP_P_TARGET)
        k_suff = min(1.0, (avail_k + rec_k) / HEMP_K_TARGET)
        nutrient_f = n_suff * 0.50 + p_suff * 0.25 + k_suff * 0.25
        yield_est = round(6.5 * ph_pen * temp_pen * drain_f * slope_f * nutrient_f, 3)

        return HempPrescriptionResult(
            rec_nitrogen_kg_ha=round(rec_n, 2),
            rec_phosphorus_kg_ha=round(rec_p, 2),
            rec_potassium_kg_ha=round(rec_k, 2),
            rec_irrigation_mm_week=round(rec_irrigation, 2),
            expected_yield_ton_ha=yield_est,
            suitable=True,
            blockers=[],
            notes=[],
            provider="rule_based",
        )


def _ph_penalty(ph: float) -> float:
    if 6.0 <= ph <= 7.0:
        return 1.0
    if ph < 6.0:
        return max(0.0, 1.0 - (6.0 - ph) * 0.35)
    return max(0.0, 1.0 - (ph - 7.0) * 0.30)


def _temp_penalty(avg_temp: float) -> float:
    if 15.0 <= avg_temp <= 27.0:
        return 1.0
    if avg_temp < 15.0:
        return max(0.0, 1.0 - (15.0 - avg_temp) * 0.06)
    return max(0.0, 1.0 - (avg_temp - 27.0) * 0.08)
