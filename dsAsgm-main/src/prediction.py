from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import HORIZONS


def selected_models(selection: str, model_names: Sequence[str]) -> list[str]:
    return list(model_names) if selection == "All Models" else [selection]


def direction_label(return_value: float, tolerance: float = 1e-12) -> str:
    if return_value > tolerance:
        return "Up"
    if return_value < -tolerance:
        return "Down"
    return "No Change"


def predict_manual(feature_row: pd.DataFrame, bundles: Mapping[str, dict], models: Sequence[str]) -> pd.DataFrame:
    current_price = float(feature_row.iloc[0]["Current_Price"])
    records = []
    for model_name in models:
        bundle = bundles[model_name]
        ordered = feature_row.loc[:, list(bundle["predictors"])]
        for horizon in HORIZONS:
            predicted_return = float(bundle["pipelines"][horizon].predict(ordered)[0])
            predicted_price = current_price * (1.0 + predicted_return)
            records.append({
                "Model": model_name,
                "Horizon": f"H{horizon}",
                "Current Price": current_price,
                "Predicted Return": predicted_return,
                "Predicted Return Percentage": predicted_return * 100.0,
                "Predicted Price Change": predicted_price - current_price,
                "Predicted Price": predicted_price,
                "Direction": direction_label(predicted_return),
            })
    return pd.DataFrame(records)


def historical_results(predictions: Mapping[str, pd.DataFrame], origin_date, models: Sequence[str]) -> pd.DataFrame:
    records = []
    selected_date = pd.Timestamp(origin_date)
    for model_name in models:
        rows = predictions[model_name].loc[predictions[model_name]["Origin_Date"].eq(selected_date)].copy()
        rows["Horizon_Number"] = rows["Horizon"].str.removeprefix("H").astype(int)
        rows = rows.sort_values("Horizon_Number")
        for row in rows.itertuples(index=False):
            predicted_return = float(row.Predicted_Return)
            records.append({
                "Model": model_name,
                "Horizon": row.Horizon,
                "Target Date": row.Target_Date,
                "Current Price": float(row.Current_Price),
                "Predicted Return": predicted_return,
                "Predicted Return Percentage": predicted_return * 100.0,
                "Predicted Price Change": float(row.Predicted_Price - row.Current_Price),
                "Predicted Price": float(row.Predicted_Price),
                "Direction": direction_label(predicted_return),
                "Actual Return (revealed after prediction)": float(row.Actual_Return),
                "Actual Price (revealed after prediction)": float(row.Actual_Price),
                "Persistence Price": float(row.Persistence_Price),
            })
    return pd.DataFrame(records)


def assert_deterministic(feature_row: pd.DataFrame, bundles: Mapping[str, dict]) -> None:
    first = predict_manual(feature_row, bundles, list(bundles))
    second = predict_manual(feature_row, bundles, list(bundles))
    numeric = ["Predicted Return", "Predicted Price"]
    if not np.array_equal(first[numeric].to_numpy(), second[numeric].to_numpy()):
        raise AssertionError("Manual deployment predictions are not deterministic.")

