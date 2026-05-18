"""
Hemp prescription synthetic dataset generator.

Produces data/hemp_training.csv with 2000 samples covering diverse soil,
field, and climate conditions. Each row encodes a complete agronomic scenario
plus the calculated optimal prescription as five regression targets.

Agronomic basis (fiber hemp, Cannabis sativa):
  pH optimal: 6.0–7.0
  N target:   ~120 kg/ha
  P target:    ~50 kg/ha
  K target:   ~100 kg/ha
  Seasonal water need: ~450 mm; prescription = deficit vs. effective rainfall
  Yield function: multiplicative penalty across pH, temperature, drainage,
                  nutrient sufficiency, and slope. Base yield 6.5 t/ha.

ppm → kg/ha conversion assumes 30 cm sampling depth, bulk density 1.3 g/cm³:
  1 ppm ≈ 3.9 kg/ha
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from random import Random

RANDOM_SEED = 42
N_SAMPLES = 2000
OUTPUT_PATH = Path("data/hemp_training.csv")

HEMP_N_TARGET_KG_HA = 120.0
HEMP_P_TARGET_KG_HA = 50.0
HEMP_K_TARGET_KG_HA = 100.0
HEMP_SEASONAL_WATER_MM = 450.0
PPM_TO_KG_HA = 3.9

DRAINAGE_CLASSES = ["poor", "moderate", "good", "excellent"]
TEXTURE_CLASSES = ["sandy loam", "loam", "silt loam", "clay loam", "silty clay loam"]


def _ph_penalty(ph: float) -> float:
    """Yield multiplier based on pH. Hemp optimal 6.0–7.0, falls off sharply outside."""
    if 6.0 <= ph <= 7.0:
        return 1.0
    if ph < 6.0:
        return max(0.0, 1.0 - (6.0 - ph) * 0.35)
    return max(0.0, 1.0 - (ph - 7.0) * 0.30)


def _temp_penalty(avg_temp: float) -> float:
    """Yield multiplier based on mean growing-season temperature. Optimal 15–27°C."""
    if 15.0 <= avg_temp <= 27.0:
        return 1.0
    if avg_temp < 15.0:
        return max(0.0, 1.0 - (15.0 - avg_temp) * 0.06)
    return max(0.0, 1.0 - (avg_temp - 27.0) * 0.08)


def _drainage_factor(drainage_class: str) -> float:
    return {"poor": 0.55, "moderate": 0.80, "good": 1.0, "excellent": 1.05}[drainage_class]


def _slope_factor(slope_percent: float) -> float:
    """Linear penalty above 8% slope."""
    return max(0.6, 1.0 - max(0.0, slope_percent - 8.0) * 0.04)


def generate_row(rng: Random, row_id: int) -> dict[str, float | int | str]:
    # ── Soil ──────────────────────────────────────────────────────────────────
    ph = round(rng.uniform(5.5, 7.8), 2)
    nitrogen_ppm = round(rng.uniform(15.0, 80.0), 1)
    phosphorus_ppm = round(rng.uniform(8.0, 45.0), 1)
    potassium_ppm = round(rng.uniform(60.0, 280.0), 1)
    organic_matter_percent = round(rng.uniform(1.0, 6.5), 2)
    ec = round(rng.uniform(0.2, 2.5), 2)
    drainage_class = rng.choice(DRAINAGE_CLASSES)
    texture_class = rng.choice(TEXTURE_CLASSES)

    # ── Field ─────────────────────────────────────────────────────────────────
    area_hectares = round(rng.uniform(2.0, 60.0), 1)
    slope_percent = round(rng.uniform(0.5, 15.0), 1)
    irrigation_available = rng.choice([True, False])
    elevation_meters = round(rng.uniform(20.0, 1100.0), 1)

    # ── Climate (seasonal averages) ───────────────────────────────────────────
    avg_temp = round(rng.uniform(10.0, 32.0), 1)
    seasonal_rainfall_mm = round(rng.uniform(150.0, 650.0), 1)
    avg_humidity = round(rng.uniform(35.0, 85.0), 1)
    avg_solar_radiation = round(rng.uniform(10.0, 24.0), 1)

    # ── Prescription rules ────────────────────────────────────────────────────
    # Available nutrient in kg/ha from soil ppm
    avail_n = nitrogen_ppm * PPM_TO_KG_HA
    avail_p = phosphorus_ppm * PPM_TO_KG_HA
    avail_k = potassium_ppm * PPM_TO_KG_HA

    # Organic matter mineralizes ~20 kg N/ha per 1% OM during the growing season
    n_from_om = organic_matter_percent * 20.0

    rec_n = max(0.0, HEMP_N_TARGET_KG_HA - avail_n - n_from_om)
    rec_p = max(0.0, HEMP_P_TARGET_KG_HA - avail_p)
    rec_k = max(0.0, HEMP_K_TARGET_KG_HA - avail_k)

    # Irrigation: water deficit relative to seasonal hemp need
    effective_rainfall = seasonal_rainfall_mm * 0.75
    raw_water_deficit = max(0.0, HEMP_SEASONAL_WATER_MM - effective_rainfall)
    irrigation_mm_week = (raw_water_deficit / 20.0) if irrigation_available else 0.0

    # ── Yield function ────────────────────────────────────────────────────────
    n_suff = min(1.0, (avail_n + n_from_om + rec_n) / HEMP_N_TARGET_KG_HA)
    p_suff = min(1.0, (avail_p + rec_p) / HEMP_P_TARGET_KG_HA)
    k_suff = min(1.0, (avail_k + rec_k) / HEMP_K_TARGET_KG_HA)
    nutrient_factor = n_suff * 0.50 + p_suff * 0.25 + k_suff * 0.25

    base_yield_ton_ha = 6.5
    raw_yield = (
        base_yield_ton_ha
        * _ph_penalty(ph)
        * _temp_penalty(avg_temp)
        * _drainage_factor(drainage_class)
        * nutrient_factor
        * _slope_factor(slope_percent)
    )

    # ── Gaussian noise ────────────────────────────────────────────────────────
    def noisy(value: float, sigma_pct: float) -> float:
        return round(max(0.0, value * (1.0 + rng.gauss(0.0, sigma_pct))), 3)

    return {
        "sample_id": row_id,
        # Soil
        "ph": ph,
        "nitrogen_ppm": nitrogen_ppm,
        "phosphorus_ppm": phosphorus_ppm,
        "potassium_ppm": potassium_ppm,
        "organic_matter_percent": organic_matter_percent,
        "ec": ec,
        "drainage_class": drainage_class,
        "texture_class": texture_class,
        # Field
        "area_hectares": area_hectares,
        "slope_percent": slope_percent,
        "irrigation_available": int(irrigation_available),
        "elevation_meters": elevation_meters,
        # Climate
        "avg_temp": avg_temp,
        "seasonal_rainfall_mm": seasonal_rainfall_mm,
        "avg_humidity": avg_humidity,
        "avg_solar_radiation": avg_solar_radiation,
        # Targets
        "rec_nitrogen_kg_ha": noisy(rec_n, 0.08),
        "rec_phosphorus_kg_ha": noisy(rec_p, 0.08),
        "rec_potassium_kg_ha": noisy(rec_k, 0.08),
        "rec_irrigation_mm_week": noisy(irrigation_mm_week, 0.10),
        "expected_yield_ton_ha": noisy(raw_yield, 0.12),
    }


def generate_dataset(n_samples: int = N_SAMPLES) -> list[dict]:
    rng = Random(RANDOM_SEED)
    return [generate_row(rng, i) for i in range(n_samples)]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_dataset()

    fieldnames = list(rows[0].keys())
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    feature_cols = [k for k in fieldnames if not k.startswith("rec_") and k not in ("sample_id", "expected_yield_ton_ha")]
    target_cols = [k for k in fieldnames if k.startswith("rec_") or k == "expected_yield_ton_ha"]
    print(f"Generated {len(rows)} rows -> {OUTPUT_PATH}")
    print(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"Targets  ({len(target_cols)}): {', '.join(target_cols)}")


if __name__ == "__main__":
    main()
