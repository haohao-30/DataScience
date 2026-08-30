from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import PREDICTORS, VISIBLE_FIELDS


EXTERNAL_FIELDS = ("USD_Index_Return_Lag1", "US10Y_Real_Yield_Change_Lag1")


def validate_manual_inputs(values: Mapping[str, float]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in VISIBLE_FIELDS if field not in values]
    if missing:
        return [f"Missing manual fields: {', '.join(missing)}"]

    numeric = {key: float(values[key]) for key in VISIBLE_FIELDS}
    if not all(np.isfinite(value) for value in numeric.values()):
        errors.append("All seven inputs must be finite numbers.")
    for field in ("Current_Price", "Current_Open", "Current_High", "Current_Low", "Price_Lag1", "Price_Lag2"):
        if numeric[field] <= 0:
            errors.append(f"{field} must be positive.")
    if numeric["Current_Volume"] < 0:
        errors.append("Current_Volume must be non-negative.")
    if numeric["Current_High"] < max(numeric["Current_Open"], numeric["Current_Low"], numeric["Current_Price"]):
        errors.append("Current_High must be at least the Open, Low and Current Price.")
    if numeric["Current_Low"] > min(numeric["Current_Open"], numeric["Current_High"], numeric["Current_Price"]):
        errors.append("Current_Low must not exceed the Open, High or Current Price.")
    return errors


def build_feature_row(
    visible_values: Mapping[str, float],
    prior_prices_before_lag2: Sequence[float],
    external_values: Mapping[str, float],
    predictor_order: Sequence[str] = PREDICTORS,
) -> pd.DataFrame:
    """Build one 22-predictor row from strictly prior/current information.

    Definitions were verified against the canonical CSV: percentage changes,
    inclusive rolling means, sample rolling standard deviations (ddof=1), and
    price momentum over 7/30 recorded observations.
    """
    errors = validate_manual_inputs(visible_values)
    if errors:
        raise ValueError(" ".join(errors))
    missing_external = [field for field in EXTERNAL_FIELDS if field not in external_values]
    if missing_external:
        raise ValueError(f"Missing stored external fields: {', '.join(missing_external)}")

    prices = np.asarray(list(prior_prices_before_lag2) + [
        float(visible_values["Price_Lag2"]),
        float(visible_values["Price_Lag1"]),
        float(visible_values["Current_Price"]),
    ], dtype=float)
    if prices.size < 31 or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("At least 28 valid earlier prices plus Lag2, Lag1 and Current Price are required.")

    price_series = pd.Series(prices, dtype=float)
    returns = price_series.pct_change()
    row = {field: float(visible_values[field]) for field in VISIBLE_FIELDS}
    row["Current_CHG"] = float(returns.iloc[-1])
    for lag in range(1, 7):
        row[f"Return_Lag{lag}"] = float(returns.iloc[-1 - lag])
    for window in (7, 30):
        row[f"MA_{window}"] = float(price_series.rolling(window).mean().iloc[-1])
        row[f"Volatility_{window}"] = float(returns.rolling(window).std(ddof=1).iloc[-1])
        row[f"Momentum_{window}"] = float(price_series.pct_change(window).iloc[-1])
    for field in EXTERNAL_FIELDS:
        row[field] = float(external_values[field])

    frame = pd.DataFrame([[row[column] for column in predictor_order]], columns=list(predictor_order))
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Constructed predictors contain a non-finite value.")
    if any(str(column).startswith("Target_") for column in frame.columns):
        raise AssertionError("Target fields must never enter the prediction row.")
    return frame


def rebuild_canonical_row(canonical: pd.DataFrame, row_index: int, predictor_order: Sequence[str] = PREDICTORS) -> pd.DataFrame:
    if row_index < 30:
        raise ValueError("Canonical parity rows need at least 30 in-file prior observations.")
    row = canonical.iloc[row_index]
    visible = {field: float(row[field]) for field in VISIBLE_FIELDS}
    external = {field: float(row[field]) for field in EXTERNAL_FIELDS}
    prior = canonical["Current_Price"].iloc[: row_index - 2].astype(float).tolist()
    return build_feature_row(visible, prior, external, predictor_order)


def feature_parity_report(canonical: pd.DataFrame, predictor_order: Sequence[str] = PREDICTORS) -> dict:
    evaluation_indices = canonical.index[canonical["Split"].eq("Evaluation")].tolist()
    indices = [60, len(canonical) // 2, evaluation_indices[0], evaluation_indices[len(evaluation_indices) // 2], evaluation_indices[-1]]
    max_error = 0.0
    for index in indices:
        rebuilt = rebuild_canonical_row(canonical, int(index), predictor_order)
        expected = canonical.loc[[index], list(predictor_order)].astype(float).reset_index(drop=True)
        difference = np.max(np.abs(rebuilt.to_numpy() - expected.to_numpy()))
        max_error = max(max_error, float(difference))
        if not np.allclose(rebuilt.to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-10):
            raise AssertionError(f"Feature parity failed at canonical row {index}; max error={difference}")
    return {"tested_indices": indices, "max_absolute_error": max_error}


def latest_manual_defaults(canonical: pd.DataFrame) -> tuple[dict[str, float], list[float], dict[str, float]]:
    latest = canonical.iloc[-1]
    visible = {field: float(latest[field]) for field in VISIBLE_FIELDS}
    # Treat the form as the next synthetic record: the two most recent stored
    # prices become Lag1/Lag2, while the latest OHLCV supplies neutral defaults.
    visible["Price_Lag1"] = float(canonical["Current_Price"].iloc[-1])
    visible["Price_Lag2"] = float(canonical["Current_Price"].iloc[-2])
    external = {field: float(latest[field]) for field in EXTERNAL_FIELDS}
    prior = canonical["Current_Price"].iloc[:-2].astype(float).tolist()
    return visible, prior, external

