"""Cross-platform, prediction-only XGBoost estimator.

It stores the already-fitted Booster in XGBoost's portable model format. It never
fits or changes the supplied trees.
"""

from __future__ import annotations

from typing import Any

import xgboost as xgb


class PortableXGBRegressor(xgb.XGBRegressor):
    def __init__(self, model_bytes: bytes, **params: Any):
        super().__init__(**params)
        self.model_bytes = bytes(model_bytes)
        self._portable_booster = None

    def get_booster(self) -> xgb.Booster:
        if self._portable_booster is None:
            booster = xgb.Booster()
            booster.load_model(bytearray(self.model_bytes))
            self._portable_booster = booster
        return self._portable_booster

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_portable_booster"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._portable_booster = None
