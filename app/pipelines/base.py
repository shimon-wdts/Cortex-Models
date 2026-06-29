from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
from prefect.context import RunContext


class Pipeline:
    def get_query_params(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return dict(parameters)

    def build_feature(self, raw_data: dict[str, pd.DataFrame], context: RunContext) -> Any:
        pass

    def run_inference(self, features: Any, context: RunContext) -> Any:
        pass

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
