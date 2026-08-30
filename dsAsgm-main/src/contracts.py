from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .constants import (
    COMPARISON_CONFIG_PATH, COMPARISON_METRICS_PATH, CONFIG_PATHS, DATA_PATH, HORIZONS,
    METRIC_COLUMNS, METRIC_PATHS, MODEL_NAMES, MODEL_PATHS, PREDICTION_COLUMNS,
    PREDICTION_PATHS, PREDICTORS, RANKING_PATH,
)
from .feature_builder import feature_parity_report, latest_manual_defaults, build_feature_row
from .prediction import assert_deterministic


def required_paths() -> list[Path]:
    return [
        DATA_PATH, *MODEL_PATHS.values(), *PREDICTION_PATHS.values(), *METRIC_PATHS.values(),
        *CONFIG_PATHS.values(), COMPARISON_METRICS_PATH, RANKING_PATH, COMPARISON_CONFIG_PATH,
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_predictors(config: dict) -> list[str]:
    value = config.get("predictors", config.get("predictor_order"))
    if value is None:
        raise AssertionError("Configuration does not contain predictors/predictor_order.")
    return list(value)


def validate_contracts(data: dict, bundles: dict[str, dict]) -> dict:
    missing = [str(path.relative_to(DATA_PATH.parents[1])) for path in required_paths() if not path.exists()]
    if missing:
        raise AssertionError(f"Required files missing: {', '.join(missing)}")

    canonical = data["canonical"]
    if canonical.shape != (3067, 45):
        raise AssertionError(f"Canonical shape is {canonical.shape}, expected (3067, 45).")
    split_counts = canonical["Split"].value_counts().to_dict()
    if split_counts != {"Train": 2452, "Evaluation": 609, "Gap": 6}:
        raise AssertionError(f"Unexpected split counts: {split_counts}")

    if set(bundles) != set(MODEL_NAMES):
        raise AssertionError("Loaded model names do not match Ridge, KNN, SVR and XGBoost.")
    for name in MODEL_NAMES:
        bundle = bundles[name]
        if bundle.get("model_name") != name:
            raise AssertionError(f"{name} joblib identifies itself as {bundle.get('model_name')!r}.")
        if list(bundle.get("horizons", [])) != list(HORIZONS):
            raise AssertionError(f"{name} horizons are not H1-H7.")
        pipelines = bundle.get("pipelines", {})
        if list(pipelines) != list(HORIZONS) or not all(isinstance(item, Pipeline) for item in pipelines.values()):
            raise AssertionError(f"{name} does not contain seven sklearn Pipelines.")
        if list(bundle.get("predictors", [])) != list(PREDICTORS):
            raise AssertionError(f"{name} predictor order differs from the required 22-column order.")

        config = data["configs"][name]
        if config.get("model_name") != name or _config_predictors(config) != list(PREDICTORS):
            raise AssertionError(f"{name} configuration contract failed.")
        if list(config.get("horizons", [])) != list(HORIZONS):
            raise AssertionError(f"{name} configuration horizons differ.")
        raw_hash = config.get("raw_input_sha256", config.get("actual_raw_input_sha256", config.get("canonical_csv_sha256")))
        if name == "KNN":
            raw_hash = config.get("raw_input_sha256")
        if raw_hash != sha256(DATA_PATH):
            raise AssertionError(f"{name} configuration refers to a different canonical CSV.")

        predictions = data["predictions"][name]
        if tuple(predictions.columns) != PREDICTION_COLUMNS:
            raise AssertionError(f"{name} prediction schema differs.")
        if len(predictions) != 4263 or not predictions.groupby("Horizon").size().eq(609).all():
            raise AssertionError(f"{name} historical prediction counts differ from 4263 / 609 per horizon.")
        metrics = data["metrics"][name]
        if tuple(metrics.columns) != METRIC_COLUMNS:
            raise AssertionError(f"{name} metric schema differs.")
        if set(metrics["Horizon"]) != {f"H{h}" for h in HORIZONS} | {"Overall"} or len(metrics) != 8:
            raise AssertionError(f"{name} metrics must contain H1-H7 plus Overall.")

    reference = data["predictions"][MODEL_NAMES[0]][["Horizon", "Origin_Date", "Target_Date"]].reset_index(drop=True)
    for name in MODEL_NAMES[1:]:
        candidate = data["predictions"][name][["Horizon", "Origin_Date", "Target_Date"]].reset_index(drop=True)
        if not reference.equals(candidate):
            raise AssertionError(f"Historical dates do not align for {name}.")
    common_dates = sorted(set(reference["Origin_Date"]))
    if len(common_dates) != 609:
        raise AssertionError(f"Expected 609 common Evaluation dates, found {len(common_dates)}.")

    ranking = data["ranking"]
    if set(ranking["Model"]) != set(MODEL_NAMES) or len(ranking) != 4:
        raise AssertionError("Comparison ranking does not contain exactly the four models.")
    best_model = data["comparison_config"].get("best_model")
    if best_model not in bundles:
        raise AssertionError(f"Comparison best_model {best_model!r} is not loaded.")
    ranked_best = ranking.sort_values("Rank").iloc[0]["Model"]
    if ranked_best != best_model:
        raise AssertionError("Comparison best_model differs from the saved unrounded ranking.")

    hash_contract = data["comparison_config"].get("input_hashes", {})
    hash_lookup = {"Canonical_CSV": DATA_PATH}
    for name in MODEL_NAMES:
        hash_lookup[f"{name}_Predictions"] = PREDICTION_PATHS[name]
        hash_lookup[f"{name}_Metrics"] = METRIC_PATHS[name]
        hash_lookup[f"{name}_Configuration"] = CONFIG_PATHS[name]
    for label, path in hash_lookup.items():
        if hash_contract.get(label) != sha256(path):
            raise AssertionError(f"Saved comparison hash mismatch for {label}.")

    parity = feature_parity_report(canonical, PREDICTORS)
    defaults, prior, external = latest_manual_defaults(canonical)
    feature_row = build_feature_row(defaults, prior, external, PREDICTORS)
    if any(column.startswith("Target_") for column in feature_row.columns):
        raise AssertionError("A target field entered the manual predictor row.")
    assert_deterministic(feature_row, bundles)
    predictions = data["predictions"][MODEL_NAMES[0]]
    reconstructed = predictions["Current_Price"] * (1.0 + predictions["Predicted_Return"])
    if not np.allclose(reconstructed, predictions["Predicted_Price"], rtol=1e-12, atol=1e-8):
        raise AssertionError("Saved predicted prices violate the reconstruction formula.")

    return {
        "best_model": best_model,
        "common_dates": common_dates,
        "feature_parity": parity,
        "canonical_sha256": sha256(DATA_PATH),
    }
