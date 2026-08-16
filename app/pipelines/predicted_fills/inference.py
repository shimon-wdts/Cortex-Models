from __future__ import annotations

import argparse
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


def build_recommendations(row: pd.Series) -> tuple[list[str], list[str], list[float], list[int], list[float], list[str], list[dict[str, object]]]:
    need_pred = int(pd.to_numeric(row.get("need_pred", 0), errors="coerce") or 0)
    need_prob = float(pd.to_numeric(row.get("need_prob", 0.0), errors="coerce") or 0.0)
    time_to_depletion = int(min(999, max(0, round((1.0 - min(need_prob, 0.999)) * 240)))) if need_pred else 999
    impact = round(need_prob * 100.0, 2)
    roi = round(max(0.0, need_prob * 10.0), 5)

    if need_pred:
        types = ["cage_fill", "table_transfer"]
        recommendations = ["Dispatch fill from cage", "Consider chip transfer between tables"]
    else:
        types = ["monitor"]
        recommendations = ["Monitor table condition"]

    modeled_impact = [impact] if len(types) == 1 else [impact, round(impact * 0.8, 2)]
    time_to_depletion_list = [time_to_depletion] * len(types)
    roi_list = [roi] if len(types) == 1 else [roi, round(roi * 0.9, 5)]
    roi_unit = ["x"] * len(types)

    rationale = []
    for idx in range(len(types)):
        pretty = (
            f"Balance is {round(float(pd.to_numeric(row.get('available_for_payout', 0.0), errors='coerce') or 0.0), 2):,.0f} above buffer | "
            f"Expected payout next 30 min: {round(float(pd.to_numeric(row.get('expected_payout_next30', 0.0), errors='coerce') or 0.0), 2):,.0f} | "
            f"EV/hr impact: {round(modeled_impact[idx], 2):,.0f} | "
            f"Safety buffer: {round(float(pd.to_numeric(row.get('safety_reserve', 0.0), errors='coerce') or 0.0), 2):,.0f}"
        )
        rationale.append(
            {
                "rel_balance": round(float(pd.to_numeric(row.get("available_for_payout", 0.0), errors="coerce") or 0.0), 2),
                "exp_payout_next30": round(float(pd.to_numeric(row.get("expected_payout_next30", 0.0), errors="coerce") or 0.0), 2),
                "exp_payout_hr": round(modeled_impact[idx], 2),
                "safety_buffer": round(float(pd.to_numeric(row.get("safety_reserve", 0.0), errors="coerce") or 0.0), 2),
                "model": row.get("model_type"),
                "pretty": pretty,
            }
        )
    return types, recommendations, modeled_impact, time_to_depletion_list, roi_list, roi_unit, rationale


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
    if action_type == "table_transfer":
        source_table_id = json_safe_value(row.get("route_v2_primary_dispatch_table_id"))
        action: dict[str, object] = {
            "type": action_type,
            "destination_table_id": table_id,
        }
        if source_table_id is not None:
            action["source_table_id"] = str(source_table_id)
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
    work = work.sort_values(["need_prob", "snapshot_ts"], ascending=[False, False], na_position="last")
    if limit is not None:
        work = work.head(limit)

    alerts: list[dict[str, object]] = []
    for _, row in work.iterrows():
        table_id = str(row.get("table_id", "unknown"))
        snapshot_ts = pd.to_datetime(row.get("snapshot_ts"), errors="coerce", utc=True)
        snapshot_text = snapshot_ts.strftime("%Y-%m-%dT%H:%M") if not pd.isna(snapshot_ts) else "unknown"
        alert_id = f"{table_id}_{snapshot_text}"
        severity = severity_from_row(row)
        types, recommendations, modeled_impact, time_to_depletion_list, roi_list, roi_unit, rationale = build_recommendations(row)
        need_prob = float(pd.to_numeric(row.get("need_prob", 0.0), errors="coerce") or 0.0)
        chart_points = [
            round(float(pd.to_numeric(row.get("tray_balance", 0.0), errors="coerce") or 0.0), 2),
            round(float(pd.to_numeric(row.get("available_for_payout", 0.0), errors="coerce") or 0.0), 2),
            round(float(pd.to_numeric(row.get("expected_payout_next30", 0.0), errors="coerce") or 0.0), 2),
            round(float(pd.to_numeric(row.get("expected_payout_next60", 0.0), errors="coerce") or 0.0), 2),
            round(float(pd.to_numeric(row.get("out_rate_per_min", 0.0), errors="coerce") or 0.0), 2),
        ]
        updated_ts = snapshot_ts if not pd.isna(snapshot_ts) else pd.Timestamp.now(tz="UTC")
        updated = updated_ts.isoformat()
        time_to_live = (updated_ts + pd.Timedelta(minutes=45)).isoformat()
        threshold = float(pd.to_numeric(row.get("decision_threshold", 0.6), errors="coerce") or 0.6)
        confidence = _confidence_from_probability(need_prob)
        expected_deficit = round(
            float(pd.to_numeric(row.get("expected_deficit_next60", 0.0), errors="coerce") or 0.0),
            2,
        )
        recommendation_items: list[dict[str, object]] = []
        for idx, action_type in enumerate(types):
            action = _fill_action(action_type, row, table_id)
            deduplication_fields = {
                "model_type": "PredictedFills",
                "table_id": table_id,
                "action_type": action_type,
                "severity": severity.lower(),
            }
            if action_type == "table_transfer" and action.get("source_table_id") is not None:
                deduplication_fields["source_table_id"] = str(action["source_table_id"])
            recommendation_items.append(
                {
                    "id": idx + 1,
                    "action": action,
                    "text": recommendations[idx],
                    "rationale": rationale[idx]["pretty"],
                    "modeled_impact": {
                        "value": modeled_impact[idx],
                        "unit": "EV/hr",
                    },
                    "roi": {
                        "value": roi_list[idx],
                        "unit": roi_unit[idx],
                    },
                    "time_to_action": {
                        "unit": "Minutes",
                        "value": time_to_depletion_list[idx],
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
            )

        alert = {
            "insights_id": f"evt_{uuid4().hex[:17].upper()}",
            "occurred_at": updated,
            "gaming_day": updated_ts.date().isoformat(),
            "source": "cortex.models.predicted_fills",
            "env": "prod",
            "version": 1.0,
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
                    "modeled_impact": {
                        "value": modeled_impact[0] if modeled_impact else 0.0,
                        "unit": "EV/hr",
                        "horizon_min": 60,
                    },
                    "expected_deficit": expected_deficit,
                    "risk_band": str(row.get("risk_band", severity.lower())),
                    "payout_risk_label": "Below buffer" if need_prob >= threshold else "Above buffer",
                    "table_state": {
                        "tray_balance": chart_points[0],
                        "available_for_payout": chart_points[1],
                        "expected_payout_next_30_min": chart_points[2],
                        "expected_payout_next_60_min": chart_points[3],
                        "out_rate_per_min": chart_points[4],
                    },
                },
                "presentation": {
                    "headline": f"Predicted fill for Table {table_id}",
                    "trigger_metric": "chip depletion probability",
                    "recommendations": recommendation_items,
                    "chart": {
                        "type": "line",
                        "y_label": "Value",
                        "x_labels": [
                            "Tray Balance",
                            "Available Payout",
                            "Expected 30m",
                            "Expected 60m",
                            "Out Rate",
                        ],
                        "series": [
                            {
                                "name": "Current",
                                "points": chart_points,
                            }
                        ],
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

