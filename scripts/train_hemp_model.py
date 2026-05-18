"""
Hemp prescription model trainer.

Reads data/hemp_training.csv, trains five independent XGBoost regressors
(one per prescription target), evaluates each on a held-out test split, and
persists artifacts to artifacts/hemp_model/.

Targets:
  rec_nitrogen_kg_ha      – nitrogen application rate
  rec_phosphorus_kg_ha    – phosphorus application rate
  rec_potassium_kg_ha     – potassium application rate
  rec_irrigation_mm_week  – weekly irrigation volume
  expected_yield_ton_ha   – forecasted fiber yield

Run:
    python scripts/train_hemp_model.py
    python scripts/train_hemp_model.py --samples 1000 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from xgboost import Booster, DMatrix, train as xgb_train
except ImportError as exc:
    raise SystemExit("xgboost is required. Install via: pip install -r requirements.txt") from exc

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "hemp_training.csv"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "hemp_model"

# ── Feature schema ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "ph",
    "nitrogen_ppm",
    "phosphorus_ppm",
    "potassium_ppm",
    "organic_matter_percent",
    "ec",
    "area_hectares",
    "slope_percent",
    "irrigation_available",
    "elevation_meters",
    "avg_temp",
    "seasonal_rainfall_mm",
    "avg_humidity",
    "avg_solar_radiation",
]

CATEGORICAL_FEATURES = [
    "drainage_class",
    "texture_class",
]

TARGETS = [
    "rec_nitrogen_kg_ha",
    "rec_phosphorus_kg_ha",
    "rec_potassium_kg_ha",
    "rec_irrigation_mm_week",
    "expected_yield_ton_ha",
]

# XGBoost hyper-parameters (shared across all targets)
XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "alpha": 0.05,
    "lambda": 1.0,
}
NUM_BOOST_ROUND = 150
TEST_FRACTION = 0.20


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TargetMetrics:
    target: str
    train_size: int
    test_size: int
    rmse: float
    mae: float
    r2: float


@dataclass
class TrainingResult:
    metrics: list[TargetMetrics]
    feature_names: list[str]
    category_levels: dict[str, list[str]]


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Feature encoding ───────────────────────────────────────────────────────────

def fit_category_levels(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Collect sorted unique values for each categorical feature."""
    levels: dict[str, list[str]] = {}
    for feature in CATEGORICAL_FEATURES:
        values = sorted({row[feature] for row in rows})
        levels[feature] = values
    return levels


def encode_row(row: dict[str, str], levels: dict[str, list[str]]) -> list[float]:
    vector: list[float] = []

    for feature in NUMERIC_FEATURES:
        vector.append(float(row[feature]))

    for feature in CATEGORICAL_FEATURES:
        value = row[feature]
        for level in levels[feature]:
            vector.append(1.0 if value == level else 0.0)

    return vector


def build_feature_names(levels: dict[str, list[str]]) -> list[str]:
    names = list(NUMERIC_FEATURES)
    for feature in CATEGORICAL_FEATURES:
        for level in levels[feature]:
            names.append(f"{feature}={level}")
    return names


def encode_all(rows: list[dict[str, str]], levels: dict[str, list[str]]) -> list[list[float]]:
    return [encode_row(row, levels) for row in rows]


# ── Train / test split ─────────────────────────────────────────────────────────

def split(rows: list, fraction: float, seed: int) -> tuple[list, list]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * fraction))
    return shuffled[cut:], shuffled[:cut]


# ── Metrics ────────────────────────────────────────────────────────────────────

def rmse(actual: list[float], predicted: list[float]) -> float:
    n = len(actual)
    return round(math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n), 4)


def mae(actual: list[float], predicted: list[float]) -> float:
    n = len(actual)
    return round(sum(abs(a - p) for a, p in zip(actual, predicted)) / n, 4)


def r2(actual: list[float], predicted: list[float]) -> float:
    mean_y = sum(actual) / len(actual)
    ss_tot = sum((a - mean_y) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    if ss_tot == 0.0:
        return 1.0
    return round(1.0 - ss_res / ss_tot, 4)


# ── Model training ─────────────────────────────────────────────────────────────

def train_one_target(
    X_train: list[list[float]],
    y_train: list[float],
    X_test: list[list[float]],
    y_test: list[float],
    feature_names: list[str],
    target_name: str,
    seed: int,
) -> tuple[Booster, TargetMetrics]:
    params = {**XGB_PARAMS, "seed": seed}
    dtrain = DMatrix(X_train, label=y_train, feature_names=feature_names)
    model = xgb_train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)

    dtest = DMatrix(X_test, feature_names=feature_names)
    preds = [float(v) for v in model.predict(dtest)]

    metrics = TargetMetrics(
        target=target_name,
        train_size=len(y_train),
        test_size=len(y_test),
        rmse=rmse(y_test, preds),
        mae=mae(y_test, preds),
        r2=r2(y_test, preds),
    )
    return model, metrics


def train_all_targets(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    X_train: list[list[float]],
    X_test: list[list[float]],
    feature_names: list[str],
    levels: dict[str, list[str]],
    seed: int,
    output_dir: Path,
) -> TrainingResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[TargetMetrics] = []

    for target in TARGETS:
        y_train = [float(row[target]) for row in train_rows]
        y_test = [float(row[target]) for row in test_rows]

        model, metrics = train_one_target(
            X_train, y_train, X_test, y_test, feature_names, target, seed
        )
        model_path = output_dir / f"{target}.json"
        model.save_model(str(model_path))
        all_metrics.append(metrics)

        print(
            f"  {target:<28}  RMSE={metrics.rmse:>7.3f}  "
            f"MAE={metrics.mae:>7.3f}  R²={metrics.r2:>6.4f}"
        )

    return TrainingResult(
        metrics=all_metrics,
        feature_names=feature_names,
        category_levels=levels,
    )


# ── Persist metadata ───────────────────────────────────────────────────────────

def save_metadata(result: TrainingResult, output_dir: Path) -> None:
    payload = {
        "targets": TARGETS,
        "feature_names": result.feature_names,
        "category_levels": result.category_levels,
        "metrics": [asdict(m) for m in result.metrics],
    }
    (output_dir / "hemp_model_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hemp prescription XGBoost models.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to training CSV.")
    parser.add_argument("--output-dir", default=str(MODEL_DIR), help="Artifact output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Training data not found: {data_path}\n"
            "Run: python scripts/generate_hemp_dataset.py"
        )

    print(f"Loading data from {data_path} …")
    rows = load_csv(data_path)
    print(f"  {len(rows)} samples loaded.")

    levels = fit_category_levels(rows)
    feature_names = build_feature_names(levels)
    print(f"  Feature vector size: {len(feature_names)}")

    train_rows, test_rows = split(rows, TEST_FRACTION, args.seed)
    print(f"  Train: {len(train_rows)}  Test: {len(test_rows)}")

    X_train = encode_all(train_rows, levels)
    X_test = encode_all(test_rows, levels)

    output_dir = Path(args.output_dir)
    print(f"\nTraining {len(TARGETS)} XGBoost models -> {output_dir}\n")

    result = train_all_targets(
        train_rows, test_rows, X_train, X_test, feature_names, levels, args.seed, output_dir
    )
    save_metadata(result, output_dir)

    print(f"\nArtifacts saved to {output_dir.resolve()}")
    print("Next: integrate HempPrescriptionProvider into app/ai/providers/ml/")


if __name__ == "__main__":
    main()
