#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .insights_contract import write_canonical_insight_outputs


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"


COHORT_COLUMNS = [
    "player_id",
    "latest_session_id",
    "period",
    "cohort_id",
    "cohort_model_label",
    "cohort_distance",
    "cohort_confidence_score",
    "target_better_cohort_id",
    "target_better_cohort_label",
    "target_better_cohort_margin",
    "cohort_edge_to_better_flag",
    "cohort_edge_score",
    "cohort_development_rank",
    "cohort_development_score",
]


RECOMMENDATION_COLUMNS = [
    "player_id",
    "latest_session_id",
    "period",
    "recommendation_candidate",
    "recommendation_candidate_reason",
    "recommended_path",
    "desired_behavior_change",
    "player_recommendation",
    "recommendation_reason",
    "supporting_signals",
    "success_metric",
    "path_fit_score",
    "recommendation_confidence",
    "cohort_id",
    "cohort_model_label",
    "target_better_cohort_id",
    "target_better_cohort_label",
    "cohort_edge_score",
    "pred_engagement_lift_prob",
    "pred_tilt_risk_prob",
    "pred_baccarat_engagement_prob",
    "pred_side_bet_engagement_prob",
    "pred_limit_path_readiness_prob",
]


def select_existing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    return df[cols]


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value)
    return default if text.lower() == "nan" else text


def rounded(value: Any, digits: int = 2) -> float:
    return round(safe_float(value), digits)


def score_0_100(value: Any) -> float:
    raw = safe_float(value)
    if raw <= 1.0:
        raw *= 100.0
    return round(max(0.0, min(100.0, raw)), 1)


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def build_cohort_json_rows(cohort_out: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in cohort_out.iterrows():
        player_id = safe_str(row.get("player_id"))
        rows.append(
            {
                "player_id": player_id,
                "latest_session_id": safe_str(row.get("latest_session_id")),
                "period": safe_str(row.get("period")),
                "cohort": {
                    "cohort_id": safe_str(row.get("cohort_id")),
                    "label": safe_str(row.get("cohort_model_label")),
                    "distance": rounded(row.get("cohort_distance"), 6),
                    "confidence_score": rounded(row.get("cohort_confidence_score"), 6),
                    "development_rank": rounded(row.get("cohort_development_rank"), 0),
                    "development_score": rounded(row.get("cohort_development_score"), 2),
                },
                "target_better_cohort": {
                    "cohort_id": safe_str(row.get("target_better_cohort_id")),
                    "label": safe_str(row.get("target_better_cohort_label")),
                    "margin": rounded(row.get("target_better_cohort_margin"), 6),
                    "edge_to_better": is_true(row.get("cohort_edge_to_better_flag")),
                    "edge_score": rounded(row.get("cohort_edge_score"), 6),
                },
            }
        )
    return rows


def build_recommendation_alert_rows(rec_out: pd.DataFrame) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    now_iso = pd.Timestamp.now("UTC").isoformat()

    for _, row in rec_out.iterrows():
        player_id = safe_str(row.get("player_id"))
        if not player_id:
            continue

        candidate = is_true(row.get("recommendation_candidate"))
        raw_path_fit = row.get("path_fit_score")
        if safe_str(raw_path_fit) == "":
            raw_path_fit = row.get("recommendation_confidence")
        path_fit = score_0_100(raw_path_fit)
        confidence = path_fit
        edge_score = score_0_100(row.get("cohort_edge_score"))
        engagement_lift = score_0_100(row.get("pred_engagement_lift_prob"))
        table_fit = score_0_100(row.get("pred_baccarat_engagement_prob"))
        limit_readiness = score_0_100(row.get("pred_limit_path_readiness_prob"))
        tilt_risk = score_0_100(row.get("pred_tilt_risk_prob"))

        recommended_path = safe_str(row.get("recommended_path"))
        recommendation = safe_str(row.get("player_recommendation"))
        reason = safe_str(row.get("recommendation_reason"))
        desired_behavior_change = safe_str(row.get("desired_behavior_change"))
        supporting_signals = safe_str(row.get("supporting_signals"))
        success_metric = safe_str(row.get("success_metric"))
        cohort_label = safe_str(row.get("cohort_model_label"))
        target_label = safe_str(row.get("target_better_cohort_label"))
        latest_session_id = safe_str(row.get("latest_session_id"))

        chart_values = [confidence, edge_score, engagement_lift, max(table_fit, limit_readiness)]
        chart_labels = ["Path Fit", "Cohort Edge", "Engagement", "Fit/Readiness"]

        if candidate:
            severity = "High" if confidence >= 75 else "Medium" if confidence >= 55 else "Low"
            alert_type = "PlayerRecommendation"
            payout_risk_label = recommended_path or "Player recommendation"
            trigger_metric = safe_str(row.get("recommendation_candidate_reason"), "Recommendation candidate")
        else:
            severity = "Info"
            alert_type = "PlayerCohortInference"
            payout_risk_label = "No player recommendation"
            trigger_metric = "Cohort inference only"

        rationale_pretty = (
            f"Path fit score: {path_fit:.0f}/100. "
            f"Cohort: {cohort_label or 'Unknown'}. "
            f"Target cohort: {target_label or 'None'}. "
            f"Reason: {reason or 'No recommendation generated.'} "
            f"Success metric: {success_metric or 'Monitor cohort movement and next-trip behavior.'}"
        )
        rationale = {
            "path_fit_score": path_fit,
            "recommendation_confidence_legacy": score_0_100(row.get("recommendation_confidence")),
            "cohort_edge_score": edge_score,
            "engagement_lift_score": engagement_lift,
            "tilt_risk_score": tilt_risk,
            "fit_readiness_score": max(table_fit, limit_readiness),
            "model": "cohort_recommendation_pipeline_v1",
            "pretty": rationale_pretty,
        }

        alert = {
            "alert_id": f"player_recommendation_{player_id}",
            "player_id": player_id,
            "session_id": latest_session_id,
            "table_id": "",
            "entity_label": f"Player {player_id}",
            "context_display": f"Player {player_id}" + (f" - Session {latest_session_id}" if latest_session_id else ""),
            "updated": now_iso,
            "alert_type": alert_type,
            "severity": severity,
            "isActionable": candidate,
            "isExportable": True,
            "payout_risk_label": payout_risk_label,
            "trigger_metric": trigger_metric,
            "types": ["player_recommendation"] if candidate else ["cohort_inference"],
            "option_count": 1 if candidate else 0,
            "modeled_impact_per_hr": [confidence],
            "time_to_depletion_min": [0],
            "roi": [confidence],
            "roi_unit": ["/100 path fit score"],
            "recommendation": [recommendation] if recommendation else [],
            "chart_series": [[chart_values]],
            "chart_labels": ["Recommendation signal"],
            "x_labels": chart_labels,
            "chart_label": "Score",
            "rationale": [rationale],
            "cohort": {
                "cohort_id": safe_str(row.get("cohort_id")),
                "label": cohort_label,
                "target_better_cohort_id": safe_str(row.get("target_better_cohort_id")),
                "target_better_cohort_label": target_label,
                "edge_score": rounded(row.get("cohort_edge_score"), 6),
            },
            "recommendation_detail": {
                "recommendation_candidate": candidate,
                "recommendation_candidate_reason": safe_str(row.get("recommendation_candidate_reason")),
                "recommended_path": recommended_path,
                "desired_behavior_change": desired_behavior_change,
                "recommendation_reason": reason,
                "supporting_signals": supporting_signals,
                "success_metric": success_metric,
                "path_fit_score": rounded(raw_path_fit, 6),
                "recommendation_confidence": rounded(row.get("recommendation_confidence"), 6),
                "pred_engagement_lift_prob": rounded(row.get("pred_engagement_lift_prob"), 6),
                "pred_tilt_risk_prob": rounded(row.get("pred_tilt_risk_prob"), 6),
                "pred_baccarat_engagement_prob": rounded(row.get("pred_baccarat_engagement_prob"), 6),
                "pred_side_bet_engagement_prob": rounded(row.get("pred_side_bet_engagement_prob"), 6),
                "pred_limit_path_readiness_prob": rounded(row.get("pred_limit_path_readiness_prob"), 6),
            },
            "exportable_content": {
                "alert_id": f"player_recommendation_{player_id}",
                "player_id": player_id,
                "session_id": latest_session_id,
                "entity_label": f"Player {player_id}",
                "alert_type": alert_type,
                "severity": severity,
                "updated": now_iso,
                "payout_risk_label": payout_risk_label,
                "trigger_metric": trigger_metric,
                "recommended_type": "player_recommendation" if candidate else "cohort_inference",
                "recommended_roi": confidence,
                "recommended_roi_unit": "/100 path fit score",
                "recommended_recommendation": recommendation,
                "recommended_rationale": rationale_pretty,
            },
        }
        alerts.append(alert)
    return alerts


def export_outputs(output_dir: Path) -> dict[str, object]:
    cohorts = pd.read_csv(output_dir / "player_cohorts.csv", low_memory=False)
    cohort_total = cohorts[cohorts["period"].eq("Total")].copy()
    cohort_out = select_existing(cohort_total, COHORT_COLUMNS)
    cohort_path = output_dir / "cohort_inference_output.csv"
    cohort_out.to_csv(cohort_path, index=False)
    cohort_json_path = output_dir / "cohort_inference_output.json"
    write_json(cohort_json_path, build_cohort_json_rows(cohort_out))

    recommendations = pd.read_csv(output_dir / "player_recommendations.csv", low_memory=False)
    rec_total = recommendations[recommendations["period"].eq("Total")].copy() if "period" in recommendations.columns else recommendations
    rec_out = select_existing(rec_total, RECOMMENDATION_COLUMNS)
    rec_path = output_dir / "recommendation_inference_output.csv"
    rec_out.to_csv(rec_path, index=False)
    recommendation_json_path = output_dir / "recommendation_alerts_output.json"
    write_json(recommendation_json_path, build_recommendation_alert_rows(rec_out))

    candidates = rec_out[rec_out["recommendation_candidate"].astype(str).str.lower().isin(["true", "1"])]
    candidate_path = output_dir / "recommendation_candidates_output.csv"
    candidates.to_csv(candidate_path, index=False)
    candidate_json_path = output_dir / "recommendation_candidate_alerts_output.json"
    write_json(candidate_json_path, build_recommendation_alert_rows(candidates))
    canonical = write_canonical_insight_outputs(output_dir)

    result = {
        "cohort_rows": int(len(cohort_out)),
        "recommendation_rows": int(len(rec_out)),
        "candidate_rows": int(len(candidates)),
        "cohort_output": str(cohort_path),
        "cohort_json_output": str(cohort_json_path),
        "recommendation_output": str(rec_path),
        "recommendation_json_output": str(recommendation_json_path),
        "candidate_output": str(candidate_path),
        "candidate_json_output": str(candidate_json_path),
    }
    result.update(canonical)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean deployable output CSVs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(export_outputs(args.output_dir))


if __name__ == "__main__":
    main()
