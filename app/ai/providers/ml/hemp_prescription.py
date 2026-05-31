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
_PPM_TO_KG_DEKAR = 0.39      # 1 ppm = 0.39 kg/dekar  (30 cm depth, 1.3 g/cm3)
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
            cat_levels = data["category_levels"]
            # Reject stale models that don't include the current categorical features
            if not all(feat in cat_levels for feat in CATEGORICAL_FEATURES):
                return False
            self._feature_names = data["feature_names"]
            self._category_levels = cat_levels
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
        n_credit  = ROTATION_N_CREDIT.get(request.previous_crop, 0.0)
        avail_n   = imputed["nitrogen_ppm"]           * _PPM_TO_KG_DEKAR
        avail_p   = imputed["phosphorus_ppm"]         * _PPM_TO_KG_DEKAR
        avail_k   = imputed["potassium_ppm"]          * _PPM_TO_KG_DEKAR
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
