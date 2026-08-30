from pathlib import Path
import ast

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.constants import (
    HORIZONS, METRIC_COLUMNS, MODEL_NAMES, MODEL_PATHS, PREDICTORS,
    PREDICTION_PATHS, ROOT, TARGET_PREFIXES,
)
from src.contracts import required_paths, validate_contracts
from src.data_access import load_all_bundles, load_all_data
from src.feature_builder import build_feature_row, feature_parity_report, latest_manual_defaults
from src.prediction import assert_deterministic, predict_manual


@pytest.fixture(scope="session")
def data():
    return load_all_data()


@pytest.fixture(scope="session")
def bundles():
    return load_all_bundles()


def test_required_files_exist():
    assert all(path.exists() for path in required_paths())


def test_all_joblibs_and_pipeline_contracts(bundles):
    assert set(bundles) == set(MODEL_NAMES)
    for name, bundle in bundles.items():
        assert bundle["model_name"] == name
        assert bundle["horizons"] == list(HORIZONS)
        assert bundle["predictors"] == list(PREDICTORS)
        assert list(bundle["pipelines"]) == list(HORIZONS)
        assert all(isinstance(pipeline, Pipeline) for pipeline in bundle["pipelines"].values())


def test_feature_builder_parity_including_evaluation(data):
    report = feature_parity_report(data["canonical"], PREDICTORS)
    assert len(report["tested_indices"]) >= 5
    assert any(data["canonical"].iloc[index]["Split"] == "Evaluation" for index in report["tested_indices"])
    assert report["max_absolute_error"] < 1e-8


def test_historical_alignment_and_counts(data):
    reference = None
    for name in MODEL_NAMES:
        predictions = data["predictions"][name]
        assert len(predictions) == 4263
        assert predictions.groupby("Horizon").size().eq(609).all()
        dates = predictions[["Horizon", "Origin_Date", "Target_Date"]].reset_index(drop=True)
        if reference is None:
            reference = dates
        else:
            pd.testing.assert_frame_equal(reference, dates)


def test_metrics_schema(data):
    for metrics in data["metrics"].values():
        assert tuple(metrics.columns) == METRIC_COLUMNS
        assert set(metrics["Horizon"]) == {f"H{h}" for h in HORIZONS} | {"Overall"}


def test_deterministic_manual_predictions_and_no_targets(data, bundles):
    visible, prior, external = latest_manual_defaults(data["canonical"])
    row = build_feature_row(visible, prior, external, PREDICTORS)
    assert not any(column.startswith(TARGET_PREFIXES) for column in row.columns)
    assert_deterministic(row, bundles)


def test_reconstructed_prices(data, bundles):
    visible, prior, external = latest_manual_defaults(data["canonical"])
    row = build_feature_row(visible, prior, external, PREDICTORS)
    results = predict_manual(row, bundles, MODEL_NAMES)
    expected = results["Current Price"] * (1.0 + results["Predicted Return"])
    assert np.allclose(expected, results["Predicted Price"], rtol=1e-12, atol=1e-8)


def test_complete_startup_contract(data, bundles):
    result = validate_contracts(data, bundles)
    assert result["best_model"] in MODEL_NAMES
    assert len(result["common_dates"]) == 609


def test_app_imports_without_running_server():
    import app
    assert callable(app.main)


def test_app_contains_no_model_fit_calls():
    paths = [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py"))]
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform"}:
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, f"Model/scaler fitting is forbidden: {violations}"


def test_paths_are_repository_relative():
    for path in [*MODEL_PATHS.values(), *PREDICTION_PATHS.values()]:
        assert path.is_relative_to(ROOT)

