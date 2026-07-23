from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from app.models.pipeline_contracts import RunContext
from app.pipelines.base import Pipeline
from app.pipelines.cohorts.cohort_inference import infer_cohorts
from app.pipelines.cohorts.insights_contract import (
    build_cohort_insight,
    build_player_score_insight,
    build_tier_lift_insight,
    merge_path_lift,
    total_period_rows,
)
from app.pipelines.cohorts.recommendation_inference import infer_recommendations
from app.pipelines.cohorts.source import CohortsFeatureResult, build_features_from_frames
from app.pipelines.cohorts.utils import safe_write_csv


class CohortsPipeline(Pipeline):
    def __init__(self, output_kind: str) -> None:
        self.output_kind = output_kind

    def get_query_params(self, parameters: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(parameters)
        source_mode = str(prepared.get("source_mode", "db")).lower()
        start_day = pd.Timestamp(prepared.get("gaming_day_start", "2026-02-01")).normalize()
        end_day = pd.Timestamp(prepared.get("gaming_day_end", "2026-02-21")).normalize()
        sample_dir = Path(__file__).resolve().parents[2] / "models_store" / "cohorts" / "samples" / "input"
        sample_path = Path(str(prepared.get("sample_feature_path") or sample_dir / "player_period_features_sample.csv"))
        prepared["gaming_day_start"] = start_day.date()
        prepared["gaming_day_end_exclusive"] = (end_day + pd.Timedelta(days=1)).date()
        prepared["limit_games"] = int(prepared.get("limit_games", 500))
        prepared["limit_bets"] = int(prepared.get("limit_bets", 5000))
        prepared["json_limit"] = int(prepared.get("json_limit", 25))
        prepared["recommend_all"] = _as_bool(prepared.get("recommend_all", True))
        prepared["source_mode"] = source_mode
        prepared["sample_feature_path"] = sample_path
        prepared["_skip_extract"] = source_mode == "sample_features"
        return prepared

    def build_feature(
        self,
        raw_data: dict[str, pd.DataFrame],
        query_parameters: dict[str, Any],
    ) -> CohortsFeatureResult:
        if query_parameters.get("source_mode") == "sample_features":
            period = pd.read_csv(Path(query_parameters["sample_feature_path"]), low_memory=False)
            return CohortsFeatureResult(
                player_period_features=period,
                player_total_features=period[period["period"].eq("Total")].copy(),
                metadata={"source_mode": "sample_features", "rows": int(len(period))},
            )

        return build_features_from_frames(
            raw_sessions=raw_data.get("sessions", pd.DataFrame()),
            raw_bets=raw_data.get("bets", pd.DataFrame()),
            raw_games=raw_data.get("games", pd.DataFrame()),
        )

    def run_inference(self, features: CohortsFeatureResult, context: RunContext) -> list[dict[str, Any]]:
        model_dir = Path(context.model_store_dir or "").resolve()
        with TemporaryDirectory(prefix=f"cortex-{context.model_name}-") as temp_name:
            output_dir = Path(temp_name)
            period_path = output_dir / "player_period_features.csv"
            total_path = output_dir / "player_total_features.csv"
            safe_write_csv(features.player_period_features, period_path)
            safe_write_csv(features.player_total_features, total_path)

            infer_cohorts(period_path, model_dir / "cohort_model.json", output_dir)
            infer_recommendations(
                output_dir / "player_cohorts.csv",
                model_dir / "recommendation_model_metadata.json",
                output_dir,
                recommend_all=bool(context.query_parameters.get("recommend_all", True)),
            )
            events = self._build_events(output_dir)

        json_limit = int(context.query_parameters.get("json_limit", 25))
        if json_limit > 0:
            return events[:json_limit]
        return events

    def _build_events(self, output_dir: Path) -> list[dict[str, Any]]:
        if self.output_kind == "cohort":
            rows = total_period_rows(pd.read_csv(output_dir / "player_cohorts.csv", low_memory=False))
            return [_with_player_key(build_cohort_insight(row)) for _, row in rows.iterrows()]

        if self.output_kind == "cohort_tier_lift":
            rows = total_period_rows(pd.read_csv(output_dir / "player_recommendations.csv", low_memory=False))
            rows = merge_path_lift(rows, output_dir)
            return [_with_player_key(build_tier_lift_insight(row)) for _, row in rows.iterrows()]

        if self.output_kind == "playerscore":
            rows = total_period_rows(pd.read_csv(output_dir / "player_total_features.csv", low_memory=False))
            return [_with_player_key(build_player_score_insight(row)) for _, row in rows.iterrows()]

        raise ValueError(f"Unsupported Cohorts output kind: {self.output_kind}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "n", "off"}


def _with_player_key(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("player_id"):
        return event
    for entity in event.get("entity", []):
        if entity.get("type") == "Player" and entity.get("id"):
            event["player_id"] = str(entity["id"])
            break
    return event
