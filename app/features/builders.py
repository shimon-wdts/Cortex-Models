from __future__ import annotations

from typing import Any

import pandas as pd


def identity_feature_builder(raw_data: pd.DataFrame, _params: dict[str, Any] | None = None) -> pd.DataFrame:
    return raw_data.copy()
