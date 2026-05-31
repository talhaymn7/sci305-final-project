"""
Hemp nested XGBoost trainer — Stage 1 classifier + Stage 2 yield regressor.

Stage 1: binary classifier (suitable = 1 / unsuitable = 0)
  - Trains on all samples from the dataset.
  - Learns which field conditions make hemp cultivation viable.

Stage 2: yield regressor (expected_yield_ton_ha)
  - Trains ONLY on samples where suitable == 1.
  - Called only when Stage 1 predicts suitable; never extrapolates to bad fields.

Artifacts saved to artifacts/hemp_model/:
  hemp_classifier.json   — XGBoost binary classifier
  hemp_regressor.json    — XGBoost yield regressor
  hemp_model_metadata.json — encoder + metrics

Run:
    python scripts/generate_hemp_dataset.py   # create data/hemp_training.csv first
    python scripts/train_hemp_model.py
    python scripts/train_hemp_model.py --samples 3000 --seed 7
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

# ── Feature schema (must match HempPrescriptionRequest field order) ────────────
NUMERIC_FEATURES = [
    "ph", "nitrogen_ppm", "phosphorus_ppm", "potassium_ppm",
    "organic_matter_percent", "ec", "area_dekar", "slope_percent",
    "first_hemp_season", "elevation_meters", "avg_temp",
    "seasonal_rainfall_mm", "avg_humidity", "avg_solar_radiation",
    "rotation_n_credit",
]
CATEGORICAL_FEATURES = ["drainage_class", "texture_class", "variety", "irrigation_type"]

# ── XGBoost hyper-parameters ──────────────────────────────────────────────────
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


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ClassifierMetrics:
    train_size: int
    test_size: int
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class RegressorMetrics:
    train_size: int
    test_size: int
    rmse: float
    mae: float
    r2: float


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Feature encoding ───────────────────────────────────────────────────────────

def fit_category_levels(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    for feature in CATEGORICAL_FEATURES:
        levels[feature] = sorted({row[feature] for row in rows})
    return levels


def build_feature_names(levels: dict[str, list[str]]) -> list[str]:
    names = list(NUMERIC_FEATURES)
    for feature in CATEGORICAL_FEATURES:
        for level in levels[feature]:
            names.append(f"{feature}={level}")
    return names


def encode_row(row: dict[str, str], levels: dict[str, list[str]]) -> list[float]:
    vector: list[float] = [float(row[f]) for f in NUMERIC_FEATURES]
    for feature in CATEGORICAL_FEATURES:
        value = row[feature]
        for level in levels[feature]:
            vector.append(1.0 if value == level else 0.0)
    return vector


def encode_all(rows: list[dict[str, str]], levels: dict[str, list[str]]) -> list[list[float]]:
    return [encode_row(row, levels) for row in rows]


# ── Train / test split ─────────────────────────────────────────────────────────

def split(rows: list, fraction: float, seed: int) -> tuple[list, list]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * fraction))
    return shuffled[cut:], shuffled[:cut]


# ── Metrics ────────────────────────────────────────────────────────────────────

def _rmse(actual: list[float], predicted: list[float]) -> float:
    n = len(actual)
    return round(math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n), 4)


def _mae(actual: list[float], predicted: list[float]) -> float:
    return round(sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual), 4)


def _r2(actual: list[float], predicted: list[float]) -> float:
    mean_y = sum(actual) / len(actual)
    ss_tot = sum((a - mean_y) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return round(1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0, 4)


def _classifier_metrics(
    y_true: list[float],
    y_prob: list[float],
    train_size: int,
    test_size: int,
) -> ClassifierMetrics:
    y_pred = [1.0 if p >= 0.5 else 0.0 for p in y_prob]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return ClassifierMetrics(
        train_size=train_size, test_size=test_size,
        accuracy=round(accuracy, 4), precision=round(precision, 4),
        recall=round(recall, 4), f1=round(f1, 4),
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train_classifier(
    X_train: list[list[float]],
    y_train: list[float],
    X_test: list[list[float]],
    y_test: list[float],
    feature_names: list[str],
    seed: int,
) -> tuple[Booster, ClassifierMetrics]:
    params = {**CLASSIFIER_PARAMS, "seed": seed}
    dtrain = DMatrix(X_train, label=y_train, feature_names=feature_names)
    model = xgb_train(params, dtrain, num_boost_round=CLASSIFIER_ROUNDS)
    dtest = DMatrix(X_test, feature_names=feature_names)
    probs = [float(v) for v in model.predict(dtest)]
    metrics = _classifier_metrics(y_test, probs, len(y_train), len(y_test))
    return model, metrics


def train_regressor(
    X_train: list[list[float]],
    y_train: list[float],
    X_test: list[list[float]],
    y_test: list[float],
    feature_names: list[str],
    seed: int,
) -> tuple[Booster, RegressorMetrics]:
    params = {**REGRESSOR_PARAMS, "seed": seed}
    dtrain = DMatrix(X_train, label=y_train, feature_names=feature_names)
    model = xgb_train(params, dtrain, num_boost_round=REGRESSOR_ROUNDS)
    dtest = DMatrix(X_test, feature_names=feature_names)
    preds = [float(v) for v in model.predict(dtest)]
    metrics = RegressorMetrics(
        train_size=len(y_train), test_size=len(y_test),
        rmse=_rmse(y_test, preds), mae=_mae(y_test, preds), r2=_r2(y_test, preds),
    )
    return model, metrics


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the hemp nested XGBoost pipeline (Stage 1 classifier + Stage 2 regressor)."
    )
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--output-dir", default=str(MODEL_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Training data not found: {data_path}\n"
            "Run: python scripts/generate_hemp_dataset.py"
        )

    print(f"Loading data from {data_path} ...")
    rows = load_csv(data_path)
    print(f"  {len(rows)} samples loaded.")

    levels = fit_category_levels(rows)
    feature_names = build_feature_names(levels)
    print(f"  Feature vector size: {len(feature_names)}")

    # Encode all features
    X_all = encode_all(rows, levels)

    train_rows, test_rows = split(rows, TEST_FRACTION, args.seed)
    X_train_all = encode_all(train_rows, levels)
    X_test_all = encode_all(test_rows, levels)
    print(f"  Train: {len(train_rows)}  Test: {len(test_rows)}")

    # ── Stage 1: Classifier ───────────────────────────────────────────────────
    y_train_cls = [float(r["suitable"]) for r in train_rows]
    y_test_cls = [float(r["suitable"]) for r in test_rows]
    n_suitable_train = int(sum(y_train_cls))
    n_unsuitable_train = len(y_train_cls) - n_suitable_train
    print(f"\n[Stage 1] Training suitability classifier ...")
    print(f"  Train labels — suitable: {n_suitable_train}  unsuitable: {n_unsuitable_train}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clf_model, clf_metrics = train_classifier(
        X_train_all, y_train_cls, X_test_all, y_test_cls, feature_names, args.seed
    )
    clf_model.save_model(str(output_dir / "hemp_classifier.json"))
    print(f"  Accuracy:  {clf_metrics.accuracy:.4f}")
    print(f"  Precision: {clf_metrics.precision:.4f}")
    print(f"  Recall:    {clf_metrics.recall:.4f}")
    print(f"  F1 Score:  {clf_metrics.f1:.4f}")

    # ── Stage 2: Regressor (suitable samples only) ────────────────────────────
    suitable_train = [(r, x) for r, x in zip(train_rows, X_train_all) if r["suitable"] == "1"]
    suitable_test = [(r, x) for r, x in zip(test_rows, X_test_all) if r["suitable"] == "1"]

    X_train_reg = [x for _, x in suitable_train]
    y_train_reg = [float(r["expected_yield_ton_dekar"]) for r, _ in suitable_train]
    X_test_reg = [x for _, x in suitable_test]
    y_test_reg = [float(r["expected_yield_ton_dekar"]) for r, _ in suitable_test]

    print(f"\n[Stage 2] Training yield regressor (suitable samples only) ...")
    print(f"  Train size: {len(X_train_reg)}  Test size: {len(X_test_reg)}")

    reg_model, reg_metrics = train_regressor(
        X_train_reg, y_train_reg, X_test_reg, y_test_reg, feature_names, args.seed
    )
    reg_model.save_model(str(output_dir / "hemp_regressor.json"))
    print(f"  RMSE: {reg_metrics.rmse:.4f} t/dekar")
    print(f"  MAE:  {reg_metrics.mae:.4f} t/dekar")
    print(f"  R²:   {reg_metrics.r2:.4f}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    metadata = {
        "feature_names": feature_names,
        "category_levels": levels,
        "classifier_metrics": asdict(clf_metrics),
        "regressor_metrics": asdict(reg_metrics),
    }
    (output_dir / "hemp_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nArtifacts saved to {output_dir.resolve()}")
    print("  hemp_classifier.json  — Stage 1 suitability classifier")
    print("  hemp_regressor.json   — Stage 2 yield regressor")
    print("  hemp_model_metadata.json")


if __name__ == "__main__":
    main()
