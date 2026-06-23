from __future__ import annotations

from typing import Any
import pandas as pd

from app.pipelines.base import Pipeline
from app.pipelines.predicted_fills.build_features import FeatureResult, FeatureConfig, build_feature_dataset
from app.pipelines.predicted_fills.fetch_data import SOURCE_TABLES, SourceBundle, df_profile, normalize_sources


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
    
    def run_inference(self, features: Any) -> Any:
        pass    


