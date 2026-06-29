from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RouteV2Config:
    urgent_threshold: float = 0.60
    opportunistic_min_prob: float = 0.30
    avoided_future_trip_minutes: float = 8.0
    same_pit_extra_stop_minutes: float = 2.0
    min_net_benefit_minutes: float = 0.0
    max_extra_stops_per_route: int = 2
    session_gap_minutes: int = 60


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _numeric(series: pd.Series | object, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series(dtype=float)


def _route_key(row: pd.Series) -> str:
    pit = str(row.get("pit_name", "") or "").strip()
    area = str(row.get("gaming_area", "") or "").strip()
    if pit and pit.lower() != "nan":
        return f"pit:{pit}"
    if area and area.lower() != "nan":
        return f"area:{area}"
    return "unknown"


def _nearest_dispatch_table(candidate: pd.Series, dispatches: pd.DataFrame) -> tuple[object | None, float | None]:
    if dispatches.empty:
        return None, None

    candidate_table = pd.to_numeric(pd.Series([candidate.get("table_id")]), errors="coerce").iloc[0]
    dispatch_table = pd.to_numeric(dispatches["table_id"], errors="coerce")
    if pd.isna(candidate_table) or dispatch_table.isna().all():
        top = dispatches.sort_values("need_prob", ascending=False).iloc[0]
        return top.get("table_id"), None

    distances = (dispatch_table - float(candidate_table)).abs()
    idx = distances.sort_values(kind="stable").index[0]
    return dispatches.loc[idx, "table_id"], float(distances.loc[idx])


def add_route_v2_recommendations(scored: pd.DataFrame, config: RouteV2Config | None = None) -> pd.DataFrame:
    cfg = config or RouteV2Config()
    out = scored.copy()
    if out.empty:
        return out

    out["snapshot_ts"] = _to_utc(out["snapshot_ts"]) if "snapshot_ts" in out.columns else pd.NaT
    out["need_prob"] = _numeric(out.get("need_prob", pd.Series(index=out.index)), 0.0)
    base_pred = _numeric(out.get("need_pred", pd.Series(index=out.index)), 0.0).astype(int)
    urgent = (base_pred == 1) | (out["need_prob"] >= cfg.urgent_threshold)

    if "score_usable_for_shadow_review" in out.columns:
        usable = out["score_usable_for_shadow_review"].astype(bool)
    else:
        usable = pd.Series(True, index=out.index)

    out["route_v2_key"] = out.apply(_route_key, axis=1)
    out["route_v2_action"] = np.where(urgent, "DISPATCH_FILL", "NO_ACTION")
    out["route_v2_pred"] = urgent.astype(int)
    out["route_v2_is_opportunistic"] = False
    out["route_v2_primary_dispatch_table_id"] = pd.NA
    out["route_v2_table_distance"] = np.nan
    out["route_v2_expected_saved_minutes"] = 0.0
    out["route_v2_extra_stop_minutes"] = 0.0
    out["route_v2_net_benefit_minutes"] = 0.0
    out["route_v2_reason"] = np.where(urgent, "urgent model threshold met", "no route action")

    eligible = (
        (~urgent)
        & usable
        & (out["need_prob"] >= cfg.opportunistic_min_prob)
        & (out["need_prob"] < cfg.urgent_threshold)
    )

    for (snapshot_ts, route_key), route in out.groupby(["snapshot_ts", "route_v2_key"], dropna=False):
        dispatches = route.loc[urgent.loc[route.index]].copy()
        if dispatches.empty:
            continue
        candidates = route.loc[eligible.loc[route.index]].copy()
        if candidates.empty:
            continue

        candidates["route_v2_expected_saved_minutes"] = candidates["need_prob"] * cfg.avoided_future_trip_minutes
        candidates["route_v2_extra_stop_minutes"] = cfg.same_pit_extra_stop_minutes
        candidates["route_v2_net_benefit_minutes"] = (
            candidates["route_v2_expected_saved_minutes"] - candidates["route_v2_extra_stop_minutes"]
        )
        candidates = candidates[candidates["route_v2_net_benefit_minutes"] >= cfg.min_net_benefit_minutes]
        candidates = candidates.sort_values(["route_v2_net_benefit_minutes", "need_prob"], ascending=[False, False])
        if cfg.max_extra_stops_per_route > 0:
            candidates = candidates.head(cfg.max_extra_stops_per_route)

        for idx, candidate in candidates.iterrows():
            primary_table, table_distance = _nearest_dispatch_table(candidate, dispatches)
            out.loc[idx, "route_v2_action"] = "ADD_TO_ROUTE"
            out.loc[idx, "route_v2_pred"] = 1
            out.loc[idx, "route_v2_is_opportunistic"] = True
            out.loc[idx, "route_v2_primary_dispatch_table_id"] = primary_table
            out.loc[idx, "route_v2_table_distance"] = table_distance
            out.loc[idx, "route_v2_expected_saved_minutes"] = float(candidate["route_v2_expected_saved_minutes"])
            out.loc[idx, "route_v2_extra_stop_minutes"] = float(candidate["route_v2_extra_stop_minutes"])
            out.loc[idx, "route_v2_net_benefit_minutes"] = float(candidate["route_v2_net_benefit_minutes"])
            out.loc[idx, "route_v2_reason"] = (
                f"same route as table {primary_table}; "
                f"prob={float(candidate['need_prob']):.3f}; "
                f"net_benefit_min={float(candidate['route_v2_net_benefit_minutes']):.2f}"
            )

    return out
