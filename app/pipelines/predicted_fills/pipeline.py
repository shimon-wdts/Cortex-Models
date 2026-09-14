from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from app.models.pipeline_contracts import RunContext
from app.pipelines.base import Pipeline
from app.pipelines.predicted_fills.build_features import FeatureResult, FeatureConfig, add_quality_flags, build_feature_dataset
from app.pipelines.predicted_fills.fetch_data import SOURCE_TABLES, SourceBundle, df_profile, normalize_sources
from app.pipelines.predicted_fills.inference import build_fill_alerts_json, score_with_saved_model
from app.pipelines.predicted_fills.route_v2 import RouteV2Config, add_route_v2_recommendations


class PredictiveFillsPipeline(Pipeline):
    def get_query_params(self, parameters: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(parameters)
        end_ts = self._coerce_timestamp(prepared.get("end_ts", "now"))
        start_offset_min = int(prepared.get("start_offset_min", 60))
        snapshot_minutes = int(prepared.get("snapshot_minute", 60))
        feature_config = FeatureConfig(snapshot_minutes=snapshot_minutes)
        start_ts = self._coerce_timestamp(
            prepared.get("start_ts") or self._shift_timestamp(end_ts, minutes=-start_offset_min)
        )

        prepared["start_ts"] = start_ts
        prepared["end_ts"] = end_ts
        prepared["snapshot_minutes"] = snapshot_minutes
        prepared["tray_history_hours"] = feature_config.stale_tray_hours
        prepared["lower"] = start_ts - pd.Timedelta(minutes=feature_config.window_min)
        prepared["tray_lower"] = start_ts - pd.Timedelta(hours=feature_config.stale_tray_hours)
        return prepared


    def build_feature(
        self,
        raw_data: dict[str, pd.DataFrame],
        query_parameters: dict[str, Any],
    ) -> FeatureResult:
        sources = normalize_sources(raw_data["tray_scans"], raw_data["chip_inventory"], raw_data["chip_updates"], raw_data["bets"])
        profile = {
            "source_tables": SOURCE_TABLES,
            "window_start": query_parameters["start_ts"],
            "window_end": query_parameters["end_ts"],
            "activity_lower_bound": query_parameters["lower"],
            "tray_scan_lower_bound": query_parameters["tray_lower"],
            "mode": "live_inference",
            "tray_scans": df_profile(sources.tray_scans, "tray_ts"),
            "chip_inventory": df_profile(sources.chip_inventory, "updated_ts"),
            "chip_updates": df_profile(sources.chip_updates, "update_ts"),
            "bets": df_profile(sources.bets, "payout_ts"),
        }

        source_bundle = SourceBundle(sources=sources, topology=raw_data["topology"], profile=profile)
        feature_result = build_feature_dataset(
            sources=source_bundle.sources,
            topology=source_bundle.topology,
            start_ts=query_parameters["start_ts"],
            end_ts=query_parameters["end_ts"],
            config=FeatureConfig(snapshot_minutes=query_parameters["snapshot_minutes"]),
        )

        return feature_result
    

    def write_outputs(self, scored: pd.DataFrame, json_limit: int, context: RunContext) -> list[dict[str, Any]]:
        scored = scored.copy()
        if "route_v2_pred" not in scored.columns:
            scored["route_v2_pred"] = pd.Series(dtype=int)

        insight_cols = [
            c for c in [
                "snapshot_ts", "table_id", "table_name", "pit_name",
                "need_prob", "need_pred", "decision_threshold", "risk_band",
                "recommended_action", "route_v2_action", "route_v2_is_opportunistic",
                "route_v2_primary_dispatch_table_id", "route_v2_net_benefit_minutes",
                "route_v2_reason",
                "tray_balance", "tray_age_hours", "OUT_total_60m", "expected_payout_next60",
                "net_buffer_next60", "denom_risk_next60_rule", "data_quality_status",
                "score_usable_for_shadow_review", "score_usable_for_auto_dispatch",
                "insight_summary",
            ]
            if c in scored.columns
        ]
        insights = scored[insight_cols].copy()
        insight_sort_cols = [c for c in ["need_prob", "snapshot_ts"] if c in insights.columns]
        if insight_sort_cols:
            insights = insights.sort_values(insight_sort_cols, ascending=[False] * len(insight_sort_cols))

        action_queue = scored.loc[pd.to_numeric(scored["route_v2_pred"], errors="coerce").fillna(0).eq(1), insight_cols].copy()
        action_sort_cols = [c for c in ["snapshot_ts", "route_v2_action", "need_prob"] if c in action_queue.columns]
        if action_sort_cols:
            ascending = [True if c != "need_prob" else False for c in action_sort_cols]
            action_queue = action_queue.sort_values(action_sort_cols, ascending=ascending)

        alerts = build_fill_alerts_json(scored, limit=json_limit, context=context)
        return alerts

    def run_inference(self, features: Any, context: RunContext) -> Any:
        model_path = Path(context.model_store_dir).resolve()
        scored = score_with_saved_model(features.features, model_dir=model_path, threshold=context.parameters["threshold"])
        scored = add_quality_flags(scored)
        route_config = RouteV2Config(
            urgent_threshold=float(context.parameters["threshold"]),
            opportunistic_min_prob=float(context.parameters["opportunistic_min_prob"]),
            avoided_future_trip_minutes=float(context.parameters["avoided_future_trip_minutes"]),
            same_pit_extra_stop_minutes=float(context.parameters["same_pit_extra_stop_minutes"]),
            max_extra_stops_per_route=int(context.parameters["max_extra_stops_per_route"]),
        )
        scored = add_route_v2_recommendations(scored, route_config)
        alerts = self.write_outputs(scored, context.parameters["json_limit"], context)
        return alerts
    


