"""
Hemp prescription synthetic dataset generator -- dekar units.

2,000 samples covering diverse soil, field, and climate conditions.
New features: variety (narli/vezir/other), irrigation_type, water_source,
previous_crop, first_hemp_season, rotation_n_credit.
Units: kg/dekar, ton/dekar.
"""
from __future__ import annotations

import csv
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
PPM_TO_KG_DEKAR         =  0.39   # 30 cm depth, 1.3 g/cm3
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
    ph                     = round(rng.uniform(5.0, 8.0), 2)
    nitrogen_ppm           = round(rng.uniform(15.0, 80.0), 1)
    phosphorus_ppm         = round(rng.uniform(8.0, 45.0), 1)
    potassium_ppm          = round(rng.uniform(60.0, 280.0), 1)
    organic_matter_percent = round(rng.uniform(1.0, 6.5), 2)
    ec                     = round(rng.uniform(0.1, 4.0), 2)
    drainage_class         = rng.choice(DRAINAGE_CLASSES)
    texture_class          = rng.choice(TEXTURE_CLASSES)

    # Field (area in dekar: 20-600 dekar = 2-60 ha)
    area_dekar       = round(rng.uniform(20.0, 600.0), 1)
    slope_percent    = round(rng.uniform(0.5, 22.0), 1)
    irrigation_type  = rng.choice(IRRIGATION_TYPES)
    water_source     = rng.choice(WATER_SOURCES)
    elevation_meters = round(rng.uniform(20.0, 1100.0), 1)

    # Variety and context
    variety           = rng.choice(VARIETIES)
    previous_crop     = rng.choice(PREVIOUS_CROPS)
    first_hemp_season = int(rng.random() < 0.4)

    # Climate
    avg_temp             = round(rng.uniform(8.0, 35.0), 1)
    seasonal_rainfall_mm = round(rng.uniform(150.0, 650.0), 1)
    avg_humidity         = round(rng.uniform(35.0, 85.0), 1)
    avg_solar_radiation  = round(rng.uniform(10.0, 24.0), 1)

    # Rotation N credit
    rotation_n_credit = ROTATION_N_CREDIT.get(previous_crop, 0.0)

    # Prescription (dekar units)
    avail_n   = nitrogen_ppm           * PPM_TO_KG_DEKAR
    avail_p   = phosphorus_ppm         * PPM_TO_KG_DEKAR
    avail_k   = potassium_ppm          * PPM_TO_KG_DEKAR
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

    # Yield
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
        "rec_nitrogen_kg_dekar":    noisy(rec_n, 0.08),
        "rec_phosphorus_kg_dekar":  noisy(rec_p, 0.08),
        "rec_potassium_kg_dekar":   noisy(rec_k, 0.08),
        "rec_irrigation_mm_week":   noisy(irrigation_mm_week, 0.10),
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
    print(f"  Unsuitable: {len(rows) - n_suitable} ({100 * (len(rows) - n_suitable) / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
