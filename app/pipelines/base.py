from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from app.models.pipeline_contracts import FeatureConfig
from app.services.imports import import_callable


class Pipeline:
    def get_query_params(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return dict(parameters)

    def build_feature(
        self,
        raw_data: dict[str, pd.DataFrame],
        feature_config: FeatureConfig,
    ) -> pd.DataFrame:
        builder = import_callable(feature_config.builder)
        features = builder(raw_data, feature_config.params)
        if not isinstance(features, pd.DataFrame):
            raise TypeError("Feature builder must return a pandas DataFrame")
        return features

    def builde_feature(
        self,
        raw_data: dict[str, pd.DataFrame],
        feature_config: FeatureConfig,
    ) -> pd.DataFrame:
        return self.build_feature(raw_data, feature_config)

    def _coerce_timestamp(self, value: Any) -> datetime:
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in {"current_time", "now", "execution_time"}:
                return datetime.now(UTC)
            try:
                return datetime.strptime(value, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=UTC)
            except ValueError:
                pass

        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(UTC)
        return timestamp.tz_convert(UTC).to_pydatetime()

    def _shift_timestamp(self, value: datetime, minutes: int) -> datetime:
        return (pd.Timestamp(value) + pd.Timedelta(minutes=minutes)).to_pydatetime()
