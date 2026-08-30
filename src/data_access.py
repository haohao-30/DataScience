from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .constants import (
    COMPARISON_CONFIG_PATH, COMPARISON_METRICS_PATH, CONFIG_PATHS, DATA_PATH,
    METRIC_PATHS, MODEL_NAMES, MODEL_PATHS, PREDICTION_PATHS, RANKING_PATH,
)


def read_csv(path: Path, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in parse_dates:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def load_all_data() -> dict:
    return {
        "canonical": read_csv(DATA_PATH, ("Origin_Date", "Target_Date", "Target_Date_H2", "Target_Date_H3", "Target_Date_H4", "Target_Date_H5", "Target_Date_H6", "Target_Date_H7")),
        "predictions": {name: read_csv(PREDICTION_PATHS[name], ("Origin_Date", "Target_Date")) for name in MODEL_NAMES},
        "metrics": {name: read_csv(METRIC_PATHS[name]) for name in MODEL_NAMES},
        "configs": {name: read_json(CONFIG_PATHS[name]) for name in MODEL_NAMES},
        "comparison_metrics": read_csv(COMPARISON_METRICS_PATH),
        "ranking": read_csv(RANKING_PATH),
        "comparison_config": read_json(COMPARISON_CONFIG_PATH),
    }


def load_all_bundles() -> dict[str, dict]:
    return {name: load_bundle(MODEL_PATHS[name]) for name in MODEL_NAMES}

