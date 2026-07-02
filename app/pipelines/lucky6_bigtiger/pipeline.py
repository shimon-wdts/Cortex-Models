from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.models.pipeline_contracts import RunContext
from app.pipelines.base import Pipeline
from app.pipelines.lucky6_bigtiger.build_features import Lucky6FeatureResult, build_feature_dataset
from app.pipelines.lucky6_bigtiger.fetch_data import SOURCE_TABLE, normalize_t_game_rows
from app.pipelines.lucky6_bigtiger.inference import score_feature_records


class Lucky6BigTigerPipeline(Pipeline):
    def get_query_params(self, parameters: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(parameters)
        end_ts = self._coerce_timestamp(prepared.get("end_ts", "now"))
        start_offset_min = int(prepared.get("start_offset_min", 5))
        start_ts = self._coerce_timestamp(
            prepared.get("start_ts") or self._shift_timestamp(end_ts, minutes=-start_offset_min)
        )

        prepared["start_ts"] = start_ts
        prepared["end_ts"] = end_ts
        prepared["source_table"] = SOURCE_TABLE
        prepared["model_set"] = str(prepared.get("model_set", "recommended"))
        prepared["hand_id_mode"] = str(prepared.get("hand_id_mode", "next"))
        prepared["json_limit"] = int(prepared.get("json_limit", 0))
        return prepared

    def build_feature(
        self,
        raw_data: dict[str, pd.DataFrame],
        query_parameters: dict[str, Any],
    ) -> Lucky6FeatureResult:
        rows = normalize_t_game_rows(raw_data.get("game_rows", pd.DataFrame()))
        return build_feature_dataset(
            rows,
            publish_start_ts=pd.Timestamp(query_parameters["start_ts"]),
            publish_end_ts=pd.Timestamp(query_parameters["end_ts"]),
            hand_id_mode=str(query_parameters.get("hand_id_mode", "next")),
        )

    def run_inference(self, features: Lucky6FeatureResult, context: RunContext) -> list[dict[str, Any]]:
        query_parameters = context.query_parameters
        json_limit = int(query_parameters.get("json_limit", 0))
        return score_feature_records(
            features,
            model_dir=Path(context.model_store_dir or "").resolve(),
            model_set=str(query_parameters.get("model_set", "recommended")),
            context=context,
            json_limit=json_limit if json_limit > 0 else None,
        )
