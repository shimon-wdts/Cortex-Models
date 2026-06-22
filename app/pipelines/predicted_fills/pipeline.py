from __future__ import annotations

from typing import Any

from app.pipelines.base import Pipeline


class PredictiveFillsPipeline(Pipeline):
    def get_query_params(self, parameters: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(parameters)
        end_ts = self._coerce_timestamp(prepared.get("end_ts", "now"))
        start_offset_min = int(prepared.get("start_offset_min", 15))
        history_window_min = int(prepared.get("history_window_min", 60))
        horizon60_min = int(prepared.get("horizon60_min", 60))
        start_ts = self._coerce_timestamp(
            prepared.get("start_ts") or self._shift_timestamp(end_ts, minutes=-start_offset_min)
        )

        prepared["start_ts"] = start_ts
        prepared["end_ts"] = end_ts
        prepared["start_offset_min"] = start_offset_min
        prepared["history_window_min"] = history_window_min
        prepared["horizon60_min"] = horizon60_min
        prepared["lower_bound"] = self._shift_timestamp(
            start_ts,
            minutes=-history_window_min,
        )
        prepared["upper_bound"] = self._shift_timestamp(
            end_ts,
            minutes=horizon60_min,
        )
        return prepared
