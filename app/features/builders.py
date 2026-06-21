from __future__ import annotations

from typing import Any

import pandas as pd


def identity_feature_builder(
    raw_data: pd.DataFrame | dict[str, pd.DataFrame],
    _params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if isinstance(raw_data, pd.DataFrame):
        return raw_data.copy()
    if len(raw_data) == 1:
        return next(iter(raw_data.values())).copy()
    return pd.concat(raw_data.values(), ignore_index=True) if raw_data else pd.DataFrame()
