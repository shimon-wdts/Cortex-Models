from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pandas as pd

from app.inference.recommendation_contract import recommendation_deduplication_id

if TYPE_CHECKING:
    from app.models.pipeline_contracts import RunContext


BASE_FEATURES = [
    "tray_balance", "available_for_payout", "safety_reserve",
    "OUT_total_60m", "bettor_out_60m", "chip_out_60m", "out_rate_per_min",
    "expected_payout_next30", "expected_payout_next60",
    "net_buffer_next30", "net_buffer_next60",
    "tray_age_hours", "minutes_since_last_seen", "tray_stale_flag",
    "expected_deficit_next30", "expected_deficit_next60",
    "denom_risk_next30_rule", "denom_risk_next60_rule",
]


def default_model_dir() -> Path:
    return Path(__file__).resolve().parent / "models" / "current_model"


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def rule_score(df: pd.DataFrame) -> pd.Series:
    deficit60 = numeric_series(df, "expected_deficit_next60").to_numpy()
    buffer30 = numeric_series(df, "net_buffer_next30").to_numpy()
    out_rate = numeric_series(df, "out_rate_per_min").to_numpy()
    denom60 = numeric_series(df, "denom_risk_next60_rule").to_numpy()
    scale = max(np.nanmedian(np.abs(buffer30)) if np.isfinite(buffer30).any() else 1.0, 1.0)
    rate_scale = max(np.nanmedian(np.abs(out_rate)) if np.isfinite(out_rate).any() else 1.0, 1e-6)
    score = (2.0 * (deficit60 / scale)) + (1.2 * (-buffer30 / scale)) + (0.8 * (out_rate / rate_scale))
    prob = sigmoid(score)
    prob = np.where(denom60.astype(int) == 1, np.maximum(prob, 0.95), prob)
    return pd.Series(np.clip(prob, 0.0, 1.0), index=df.index, name="need_prob")


def load_model_artifacts(model_dir: Path) -> tuple[dict, object | None]:
    meta_path = model_dir / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = model_dir / "model.pkl"
    estimator = None
    if model_path.exists():
        try:
            with model_path.open("rb") as fh:
                estimator = pickle.load(fh)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency while loading saved model. Install dependencies from requirements.txt."
            ) from exc
    return metadata, estimator


def score_with_saved_model(features: pd.DataFrame, model_dir: Path | None = None, threshold: float = 0.60) -> pd.DataFrame:
    resolved_model_dir = model_dir or default_model_dir()
    out = features.copy()
    if out.empty:
        meta_path = resolved_model_dir / "model_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing model metadata: {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        out["need_prob"] = pd.Series(dtype=float)
        out["need_pred"] = pd.Series(dtype=int)
        out["decision_threshold"] = pd.Series(dtype=float)
        out["model_type"] = str(metadata.get("model_type", "unknown"))
        out["recommended_action"] = pd.Series(dtype=str)
        out["risk_band"] = pd.Series(dtype=str)
        out["insight_summary"] = pd.Series(dtype=str)
        return out
    metadata, estimator = load_model_artifacts(resolved_model_dir)
    feature_cols = metadata.get("feature_cols", [])
    if estimator is not None and feature_cols:
        X = out.reindex(columns=feature_cols, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["need_prob"] = estimator.predict_proba(X)[:, 1]
    else:
        out["need_prob"] = rule_score(out)
    out["need_pred"] = (out["need_prob"] >= threshold).astype(int)
    out["decision_threshold"] = float(threshold)
    out["model_type"] = str(metadata.get("model_type", "unknown"))
    out["recommended_action"] = out["need_pred"].map({1: "DISPATCH_FILL", 0: "NO_ACTION"})
    out["risk_band"] = pd.cut(
        out["need_prob"],
        bins=[-0.001, 0.30, 0.60, 0.85, 1.0],
        labels=["low", "watch", "high", "critical"],
    ).astype(str)
    out["insight_summary"] = out.apply(
        lambda row: f"table {row.get('table_id', 'unknown')} probability={float(row.get('need_prob', 0.0)):.3f} action={row.get('recommended_action', 'UNKNOWN')}",
        axis=1,
    )
    return out


def json_safe_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def severity_from_row(row: pd.Series) -> str:
    prob = float(pd.to_numeric(row.get("need_prob", 0.0), errors="coerce") or 0.0)
    if prob >= 0.85:
        return "Critical"
    if prob >= 0.60:
        return "High"
    if prob >= 0.30:
        return "Medium"
    return "Low"


def _numeric_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = pd.to_numeric(row.get(key, default), errors="coerce")
    return default if pd.isna(value) else float(value)


def _optional_numeric_value(row: pd.Series, key: str) -> float | None:
    value = pd.to_numeric(row.get(key), errors="coerce")
    return None if pd.isna(value) else float(value)


def _time_to_depletion_minutes(row: pd.Series) -> int | None:
    candidates: list[float] = []
    available = max(0.0, _numeric_value(row, "available_for_payout"))
    out_rate = max(0.0, _numeric_value(row, "out_rate_per_min"))
    if out_rate > 0:
        candidates.append(available / out_rate)

    denomination_minutes = _optional_numeric_value(row, "worst_denom_minutes_to_zero")
    if denomination_minutes is not None and denomination_minutes >= 0:
        candidates.append(denomination_minutes)

    if not candidates:
        return None
    return max(0, int(round(min(candidates))))


def _table_display_name(row: pd.Series, table_id: str) -> str:
    table_name = str(row.get("table_name", "") or "").strip()
    return table_name if table_name and table_name.lower() != "nan" else f"Table {table_id}"


def _table_context(row: pd.Series, display_name: str) -> str:
    parts = [display_name]
    for key in ("pit_name", "gaming_area"):
        value = str(row.get(key, "") or "").strip()
        if value and value.lower() != "nan" and value not in parts:
            parts.append(value)
    return " | ".join(parts)


def _estimated_theo_impact(
    row: pd.Series,
    need_probability: float,
    downtime_minutes_avoided: float,
) -> dict[str, object]:
    average_theo_per_hour = max(0.0, _numeric_value(row, "theo_last_60m"))
    minutes_avoided = max(0.0, downtime_minutes_avoided)
    expected_theo_preserved = average_theo_per_hour * (minutes_avoided / 60.0) * need_probability
    return {
        "value": round(expected_theo_preserved, 2),
        "unit": "theo",
        "label": "Estimated theo preserved",
        "average_theo_per_hour": round(average_theo_per_hour, 2),
        "downtime_minutes_avoided": round(minutes_avoided, 2),
        "fill_need_probability": round(need_probability, 6),
        "calculation": "average_theo_per_hour * downtime_minutes_avoided / 60 * fill_need_probability",
    }


def _confidence_from_probability(probability: float) -> str:
    if probability >= 0.85:
        return "high"
    if probability >= 0.60:
        return "medium"
    return "low"


def _fill_action(action_type: str, row: pd.Series, table_id: str) -> dict[str, object]:
    if action_type == "cage_fill":
        return {
            "type": action_type,
            "source": "cage",
            "destination_table_id": table_id,
        }
    if action_type == "add_to_route":
        primary_dispatch_table_id = json_safe_value(row.get("route_v2_primary_dispatch_table_id"))
        action: dict[str, object] = {
            "type": action_type,
            "destination_table_id": table_id,
        }
        if primary_dispatch_table_id is not None:
            action["primary_dispatch_table_id"] = str(primary_dispatch_table_id)
        return action
    return {
        "type": action_type,
        "metric": "chip_depletion_risk",
        "table_id": table_id,
    }


def build_fill_alerts_json(
    scored: pd.DataFrame,
    limit: int | None = None,
    context: "RunContext | None" = None,
) -> list[dict[str, object]]:
    work = scored.copy()
    if "snapshot_ts" in work.columns:
        work["snapshot_ts"] = pd.to_datetime(work["snapshot_ts"], errors="coerce", utc=True)
    sort_columns = [column for column in ("need_prob", "snapshot_ts") if column in work.columns]
    if sort_columns:
        work = work.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
    if limit is not None:
        work = work.head(limit)

    alerts: list[dict[str, object]] = []
    for _, row in work.iterrows():
        table_id = str(row.get("table_id", "unknown"))
        display_name = _table_display_name(row, table_id)
        table_context = _table_context(row, display_name)
        snapshot_ts = pd.to_datetime(row.get("snapshot_ts"), errors="coerce", utc=True)
        snapshot_text = snapshot_ts.strftime("%Y-%m-%dT%H:%M") if not pd.isna(snapshot_ts) else "unknown"
        alert_id = f"{table_id}_{snapshot_text}"
        severity = severity_from_row(row)
        need_pred = int(_numeric_value(row, "need_pred"))
        need_prob = min(1.0, max(0.0, _numeric_value(row, "need_prob")))
        time_to_depletion = _time_to_depletion_minutes(row)
        configured_downtime_minutes = 5.0
        if context is not None:
            configured_downtime_minutes = float(
                context.parameters.get("estimated_downtime_minutes_avoided", configured_downtime_minutes)
            )
        modeled_impact = _estimated_theo_impact(row, need_prob, configured_downtime_minutes)

        tray_balance = round(_numeric_value(row, "tray_balance"), 2)
        safety_reserve = round(_numeric_value(row, "safety_reserve"), 2)
        available_for_payout = round(_numeric_value(row, "available_for_payout"), 2)
        expected_payout_30 = round(_numeric_value(row, "expected_payout_next30"), 2)
        expected_payout_60 = round(_numeric_value(row, "expected_payout_next60"), 2)
        out_rate_per_min = round(max(0.0, _numeric_value(row, "out_rate_per_min")), 2)
        net_buffer_30 = round(available_for_payout - expected_payout_30, 2)
        net_buffer_60 = round(available_for_payout - expected_payout_60, 2)

        projection_minutes = [0, 10, 20, 30, 40, 50, 60]
        projected_buffer = [
            round(available_for_payout - (out_rate_per_min * minute), 2)
            for minute in projection_minutes
        ]
        updated_ts = snapshot_ts if not pd.isna(snapshot_ts) else pd.Timestamp.now(tz="UTC")
        updated = updated_ts.isoformat()
        time_to_live = (updated_ts + pd.Timedelta(minutes=45)).isoformat()
        threshold = _numeric_value(row, "decision_threshold", 0.6)
        confidence = _confidence_from_probability(need_prob)
        expected_deficit = round(max(0.0, _numeric_value(row, "expected_deficit_next60", -net_buffer_60)), 2)

        route_action = str(row.get("route_v2_action", "") or "").upper()
        if need_pred:
            action_type = "cage_fill"
            deadline = f" within {time_to_depletion} minutes" if time_to_depletion is not None else ""
            recommendation_text = f"Dispatch a chip fill to {display_name}{deadline}."
        elif route_action == "ADD_TO_ROUTE":
            action_type = "add_to_route"
            recommendation_text = f"Add {display_name} to the active chip-fill route."
        else:
            action_type = "monitor"
            recommendation_text = f"Monitor {display_name}; no immediate fill is required."

        if action_type == "monitor":
            modeled_impact = {
                **modeled_impact,
                "value": 0.0,
                "label": "Estimated theo preserved by monitoring",
            }

        if need_pred and time_to_depletion is not None:
            trigger_metric = (
                "Available chips above the safety reserve are projected to run out "
                f"in approximately {time_to_depletion} minutes."
            )
        elif need_pred:
            trigger_metric = "Expected payouts exceed the available chip balance within the next 60 minutes."
        elif action_type == "add_to_route":
            trigger_metric = "This table can be serviced efficiently as part of an active chip-fill route."
        else:
            trigger_metric = "Projected chip balance remains above the safety reserve for the next 60 minutes."

        balance_rationale = (
            f"{available_for_payout:,.0f} is available above the {safety_reserve:,.0f} safety reserve, "
            f"compared with {expected_payout_60:,.0f} in expected payouts over the next 60 minutes."
        )
        if action_type == "monitor":
            rationale = f"{balance_rationale} No immediate fill is expected to preserve additional theo."
        else:
            rationale = (
                f"{balance_rationale} Acting proactively is estimated to avoid "
                f"{modeled_impact['downtime_minutes_avoided']:g} minutes of downtime and preserve "
                f"{modeled_impact['value']:,.2f} in theo."
            )

        action = _fill_action(action_type, row, table_id)
        deduplication_fields = {
            "model_type": "PredictedFills",
            "table_id": table_id,
            "action_type": action_type,
            "severity": severity.lower(),
        }
        if action_type == "add_to_route" and action.get("primary_dispatch_table_id") is not None:
            deduplication_fields["primary_dispatch_table_id"] = str(action["primary_dispatch_table_id"])
        recommendation_items = [
            {
                "id": 1,
                "action": action,
                "text": recommendation_text,
                "context": table_context,
                "trigger": trigger_metric,
                "expected_value": modeled_impact,
                "rationale": rationale,
                "modeled_impact": modeled_impact,
                "time_to_action": {
                    "unit": "Minutes",
                    "value": time_to_depletion,
                },
                "confidence": confidence,
                "time_to_live": time_to_live,
                "thresholds": [
                    {
                        "metric": "payload.result.score",
                        "operator": ">=",
                        "value": threshold,
                    }
                ],
                "deduplication_id": recommendation_deduplication_id(deduplication_fields),
            }
        ]

        if action_type in {"cage_fill", "add_to_route"}:
            recommended_label = "Dispatch now" if action_type == "cage_fill" else "Add to route"
            action_comparison = [
                {
                    "id": action_type,
                    "label": recommended_label,
                    "recommended": True,
                    "expected_outcome": (
                        f"Avoid approximately {modeled_impact['downtime_minutes_avoided']:g} minutes of downtime."
                    ),
                    "expected_theo_preserved": modeled_impact["value"],
                },
                {
                    "id": "wait",
                    "label": "Wait",
                    "recommended": False,
                    "expected_outcome": (
                        f"Risk chip depletion in approximately {time_to_depletion} minutes."
                        if time_to_depletion is not None
                        else "Risk chip depletion within the next 60 minutes."
                    ),
                    "expected_theo_preserved": 0.0,
                },
            ]
        else:
            action_comparison = [
                {
                    "id": "monitor",
                    "label": "Monitor",
                    "recommended": True,
                    "expected_outcome": "No immediate fill; continue monitoring the projected chip buffer.",
                    "expected_theo_preserved": 0.0,
                },
                {
                    "id": "dispatch_now",
                    "label": "Dispatch now",
                    "recommended": False,
                    "expected_outcome": "No modeled downtime benefit at this time.",
                    "expected_theo_preserved": 0.0,
                },
            ]

        alert = {
            "insights_id": f"evt_{uuid4().hex[:17].upper()}",
            "occurred_at": updated,
            "gaming_day": updated_ts.date().isoformat(),
            "source": "cortex.models.predicted_fills",
            "env": "prod",
            "version": 2.0,
            "model": {
                "type": "PredictedFills",
                "version": str(
                    context.parameters.get("model_version", "10.2.0")
                    if context is not None
                    else "10.2.0"
                ),
                "feature_version": str(
                    context.parameters.get("feature_version", "1.0")
                    if context is not None
                    else "1.0"
                ),
                "run_id": context.run_id if context is not None else None,
            },
            "entity": [
                {
                    "type": "TABLE",
                    "id": table_id,
                    "present_in_user_interface": True,
                }
            ],
            "severity": severity.lower(),
            "application": ["cortexFloor"],
            "alert_id": alert_id,
            "table_id": table_id,
            "payload": {
                "result": {
                    "decision_class": "chip_depletion",
                    "score": round(need_prob, 6),
                    "confidence": confidence,
                    "modeled_impact": {**modeled_impact, "horizon_min": 60},
                    "expected_deficit": expected_deficit,
                    "risk_band": str(row.get("risk_band", severity.lower())),
                    "payout_risk_label": "Below buffer" if need_prob >= threshold else "Above buffer",
                    "table_state": {
                        "table_id": table_id,
                        "table_name": display_name,
                        "tray_balance": tray_balance,
                        "safety_reserve": safety_reserve,
                        "available_for_payout": available_for_payout,
                        "expected_payout_next_30_min": expected_payout_30,
                        "expected_payout_next_60_min": expected_payout_60,
                        "net_buffer_next_30_min": net_buffer_30,
                        "net_buffer_next_60_min": net_buffer_60,
                        "out_rate_per_min": out_rate_per_min,
                        "average_theo_per_hour": modeled_impact["average_theo_per_hour"],
                        "time_to_depletion_min": time_to_depletion,
                    },
                },
                "presentation": {
                    "message_type": "Predicted Fill Need",
                    "message_type_code": "PREDICTED_FILL_NEED",
                    "headline": (
                        f"Predicted chip fill needed at {display_name}"
                        if need_pred
                        else (
                            f"Add {display_name} to the active fill route"
                            if action_type == "add_to_route"
                            else f"Monitor chip levels at {display_name}"
                        )
                    ),
                    "recommendation": recommendation_text,
                    "context": table_context,
                    "trigger_metric": trigger_metric,
                    "expected_value": modeled_impact,
                    "rationale": rationale,
                    "recommendations": recommendation_items,
                    "action_comparison": action_comparison,
                    "chart": {
                        "type": "line",
                        "title": "Projected chips above safety reserve",
                        "y_label": "Chips above safety reserve",
                        "x_label": "Minutes from now",
                        "x_labels": ["Now", "+10m", "+20m", "+30m", "+40m", "+50m", "+60m"],
                        "series": [
                            {
                                "name": "Wait / no fill",
                                "points": projected_buffer,
                            },
                            {
                                "name": "Safety reserve threshold",
                                "points": [0.0] * len(projection_minutes),
                            },
                        ],
                        "annotations": (
                            [
                                {
                                    "type": "vertical_marker",
                                    "x_value_min": time_to_depletion,
                                    "label": "Projected depletion",
                                }
                            ]
                            if time_to_depletion is not None
                            else []
                        ),
                    },
                },
                "actions": {
                    "available": ["approve", "decline", "export"],
                    "default": "archive",
                    "requires_reason_on": ["decline"],
                    "export": {
                        "formats": ["pdf", "csv"],
                        "scope": "recommendation",
                    },
                },
            },
        }
        alerts.append({k: json_safe_value(v) if not isinstance(v, (list, dict)) else v for k, v in alert.items()})
    return alerts

