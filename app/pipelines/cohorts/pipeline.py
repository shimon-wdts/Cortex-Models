from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.models.pipeline_contracts import RunContext
from app.pipelines.base import Pipeline
from app.pipelines.cohorts.cohort_inference import infer_cohorts_from_frame
from app.pipelines.cohorts.config import MIN_BETS
from app.pipelines.cohorts.insights_contract import (
    build_cohort_insight,
    build_player_score_insight,
    build_tier_lift_insight,
    total_period_rows,
)
from app.pipelines.cohorts.recommendation_inference import infer_recommendations_from_frame
from app.pipelines.cohorts.source import CohortsFeatureResult, build_features_from_frames


class CohortsPipeline(Pipeline):
    def __init__(self, output_kind: str) -> None:
        self.output_kind = output_kind

    def get_query_params(self, parameters: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(parameters)
        source_mode = str(prepared.get("source_mode", "db")).lower()
        end_day = _gaming_day(prepared.get("gaming_day_end", "now"))
        observation_days = max(1, int(prepared.get("observation_days", 21)))
        start_value = prepared.get("gaming_day_start")
        start_day = _gaming_day(start_value) if start_value else end_day - timedelta(days=observation_days - 1)
        if start_day > end_day:
            raise ValueError("gaming_day_start must be on or before gaming_day_end")

        sample_dir = Path(__file__).resolve().parents[2] / "models_store" / "cohorts" / "samples" / "input"
        sample_path = Path(str(prepared.get("sample_feature_path") or sample_dir / "player_period_features_sample.csv"))
        prepared["gaming_day_start"] = start_day.date()
        prepared["gaming_day_end_exclusive"] = (end_day + timedelta(days=1)).date()
        prepared["observation_days"] = int((end_day - start_day).days) + 1
        prepared["limit_games"] = _optional_limit(prepared.get("limit_games"))
        prepared["limit_bets"] = _optional_limit(prepared.get("limit_bets"))
        prepared["json_limit"] = int(prepared.get("json_limit", 0))
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
            observation_start=query_parameters["gaming_day_start"],
        )

    def run_inference(self, features: CohortsFeatureResult, context: RunContext) -> list[dict[str, Any]]:
        features = _eligible_feature_result(features)
        if features.player_total_features.empty:
            return []

        model_dir = Path(context.model_store_dir or "").resolve()
        cohorts = infer_cohorts_from_frame(
            features.player_period_features,
            model_dir / "cohort_model.json",
        )
        recommendations = infer_recommendations_from_frame(
            cohorts,
            model_dir / "recommendation_model_metadata.json",
            recommend_all=bool(context.query_parameters.get("recommend_all", True)),
        )
        events = self._build_events(features, cohorts, recommendations)

        json_limit = int(context.query_parameters.get("json_limit", 0))
        if json_limit > 0:
            return events[:json_limit]
        return events

    def _build_events(
        self,
        features: CohortsFeatureResult,
        cohorts: pd.DataFrame,
        recommendations: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        if self.output_kind == "cohort":
            rows = total_period_rows(cohorts)
            return [_with_player_key(build_cohort_insight(row)) for _, row in rows.iterrows()]

        if self.output_kind == "cohort_tier_lift":
            rows = total_period_rows(recommendations)
            return [_with_player_key(build_tier_lift_insight(row)) for _, row in rows.iterrows()]

        if self.output_kind == "playerscore":
            rows = total_period_rows(features.player_total_features)
            return [_with_player_key(build_player_score_insight(row)) for _, row in rows.iterrows()]

        raise ValueError(f"Unsupported Cohorts output kind: {self.output_kind}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "n", "off"}


def _gaming_day(value: Any) -> pd.Timestamp:
    if value is None or str(value).strip().lower() == "now":
        return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    day = pd.Timestamp(value)
    if day.tzinfo is not None:
        day = day.tz_convert("UTC").tz_localize(None)
    return day.normalize()


def _optional_limit(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    limit = int(value)
    return limit if limit > 0 else None


def _eligible_feature_result(features: CohortsFeatureResult) -> CohortsFeatureResult:
    totals = total_period_rows(features.player_total_features).copy()
    if totals.empty:
        return CohortsFeatureResult(
            player_period_features=features.player_period_features.iloc[0:0].copy(),
            player_total_features=totals,
            metadata=dict(features.metadata),
        )

    eligible = pd.Series(True, index=totals.index, dtype=bool)
    if "eligible" in totals.columns:
        values = totals["eligible"]
        if pd.api.types.is_bool_dtype(values):
            eligible &= values.fillna(False)
        else:
            eligible &= values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    if "bet_count" in totals.columns:
        eligible &= pd.to_numeric(totals["bet_count"], errors="coerce").fillna(0).ge(MIN_BETS)

    eligible_totals = totals.loc[eligible].copy()
    period = features.player_period_features
    if "player_id" in period.columns and "player_id" in eligible_totals.columns:
        eligible_ids = set(eligible_totals["player_id"].astype(str))
        eligible_period = period.loc[period["player_id"].astype(str).isin(eligible_ids)].copy()
    else:
        eligible_period = period.copy()

    metadata = {
        **features.metadata,
        "minimum_bets": MIN_BETS,
        "eligible_player_rows": int(len(eligible_totals)),
    }
    return CohortsFeatureResult(
        player_period_features=eligible_period,
        player_total_features=eligible_totals,
        metadata=metadata,
    )


def _with_player_key(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("player_id"):
        return event
    for entity in event.get("entity", []):
        if entity.get("type") == "Player" and entity.get("id"):
            event["player_id"] = str(entity["id"])
            break
    return event
