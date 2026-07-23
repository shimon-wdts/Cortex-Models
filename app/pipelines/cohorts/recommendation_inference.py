#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, IN_COHORT_OPTIMIZATION_CONFIG
from .utils import enable_local_catboost, read_json, safe_write_csv, write_json


def numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    values = df[col]
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def safe_num(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def add_recommendation_candidate_flags(scored: pd.DataFrame) -> pd.DataFrame:
    """Add all priority candidate gates used by inference and validation."""
    scored = scored.copy()
    config = IN_COHORT_OPTIMIZATION_CONFIG
    max_rank = numeric_series(scored, "cohort_development_rank").max()
    lowest_threshold = max(1.0, max_rank - 1.0)

    edge_flag = bool_series(scored, "cohort_edge_to_better_flag")
    lowest_flag = numeric_series(scored, "cohort_development_rank").ge(lowest_threshold)
    scored["lowest_development_cohort_flag"] = lowest_flag

    distance = numeric_series(scored, "cohort_distance")
    scored["cohort_distance_pct_in_cohort"] = (
        scored.assign(_cohort_distance=distance)
        .groupby("cohort_id")["_cohort_distance"]
        .rank(method="average", pct=True)
        .fillna(0.0)
    )

    opportunity_score = pd.concat(
        [
            numeric_series(scored, "hidden_opportunity_score"),
            numeric_series(scored, "table_fit_score"),
            numeric_series(scored, "rhythm_score"),
            numeric_series(scored, "worth_score"),
            numeric_series(scored, "frequency_score"),
            numeric_series(scored, "pred_engagement_lift_prob") * 100.0,
        ],
        axis=1,
    ).max(axis=1)
    scored["in_cohort_opportunity_score"] = opportunity_score
    scored["in_cohort_optimization_score"] = (
        float(config["opportunity_weight"]) * (opportunity_score / 100.0).clip(0.0, 1.0)
        + float(config["distance_weight"]) * numeric_series(scored, "cohort_distance_pct_in_cohort").clip(0.0, 1.0)
    )
    scored["in_cohort_optimization_flag"] = (
        ~edge_flag
        & ~lowest_flag
        & numeric_series(scored, "cohort_development_rank").le(max_rank - float(config["max_rank_offset_from_lowest"]))
        & numeric_series(scored, "cohort_distance_pct_in_cohort").ge(float(config["min_distance_pct_in_cohort"]))
        & numeric_series(scored, "in_cohort_opportunity_score").ge(float(config["min_opportunity_score"]))
        & numeric_series(scored, "pred_engagement_lift_prob").ge(float(config["min_engagement_lift_prob"]))
    )

    in_cohort_flag = bool_series(scored, "in_cohort_optimization_flag")
    scored["priority_recommendation_candidate"] = edge_flag | lowest_flag | in_cohort_flag
    scored["recommendation_candidate"] = scored["priority_recommendation_candidate"]
    scored["recommendation_candidate_reason"] = np.select(
        [edge_flag, lowest_flag, in_cohort_flag],
        [
            "On the edge of a higher-development cohort",
            "In a lowest-development cohort",
            "In-cohort optimization opportunity",
        ],
        default="No recommendation candidate",
    )
    return scored


def build_trend_context(data: pd.DataFrame) -> pd.DataFrame:
    weeks = data[data["period"].isin(["Week 1", "Week 2", "Week 3"])].copy()
    cols = [
        "active_days",
        "hours_played",
        "num_sessions",
        "chase_rate",
        "volatility_score_behavior",
        "baccarat_engagement_pct",
        "blackjack_engagement_pct",
        "side_bet_rate",
        "ceiling_pressure_score",
        "range_width_score",
        "stretch_capacity_score",
        "rhythm_score",
        "session_fade_score",
        "post_loss_response_score",
        "game_affinity_score",
        "table_fit_score",
        "hidden_opportunity_score",
        "confidence_need_score",
    ]
    present = [c for c in cols if c in weeks.columns]
    if not present:
        return pd.DataFrame({"player_id": data["player_id"].drop_duplicates()})
    pivot = weeks.pivot_table(index="player_id", columns="period", values=present, aggfunc="first")
    out = pd.DataFrame(index=pivot.index)
    for col in present:
        w1 = pivot.get((col, "Week 1"))
        w3 = pivot.get((col, "Week 3"))
        if w1 is not None and w3 is not None:
            out[f"{col}_w3_vs_w1"] = w3.fillna(0.0) - w1.fillna(0.0)
    return out.reset_index()


def load_predictions(total: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    enable_local_catboost()
    from catboost import CatBoostClassifier, Pool

    features = list(metadata.get("features", []))
    cat_features = list(metadata.get("categorical_features", []))
    pred = total.copy()
    for col in features:
        if col in cat_features:
            pred[col] = pred[col].fillna("Unknown").astype(str) if col in pred.columns else "Unknown"
        else:
            pred[col] = pd.to_numeric(pred[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) if col in pred.columns else 0.0

    cat_idx = [features.index(c) for c in cat_features if c in features]
    pool = Pool(pred[features], cat_features=cat_idx)
    model_names = list(metadata.get("models", {}).keys())
    for name in model_names:
        info = metadata.get("models", {}).get(name, {})
        col = f"pred_{name}_prob"
        if info.get("status") != "trained" or not info.get("path"):
            pred[col] = 0.0
            continue
        model = CatBoostClassifier()
        model_path = Path(str(info["path"]))
        if not model_path.is_absolute():
            model_path = Path(str(metadata.get("_metadata_dir", ""))) / model_path
        model.load_model(model_path)
        pred[col] = model.predict_proba(pool)[:, 1]
    return pred


def choose_recommendation(row: pd.Series) -> dict[str, object]:
    if not bool(row.get("recommendation_candidate", False)):
        return {
            "recommended_path": "",
            "desired_behavior_change": "",
            "recommendation_action": "",
            "recommendation_target": "",
            "player_recommendation": "",
            "recommendation_reason": "No recommendation generated: player is not on the edge of a better cohort, in a lowest-development cohort, or a strong in-cohort optimization opportunity.",
            "supporting_signals": "",
            "success_metric": "",
            "recommendation_confidence": 0.0,
        }

    tilt_prob = safe_num(row, "pred_tilt_risk_prob")
    engagement_prob = safe_num(row, "pred_engagement_lift_prob")
    baccarat_prob = safe_num(row, "pred_baccarat_engagement_prob")
    side_prob = safe_num(row, "pred_side_bet_engagement_prob")
    limit_prob = safe_num(row, "pred_limit_path_readiness_prob")
    chase = safe_num(row, "chase_rate")
    volatility = safe_num(row, "volatility_score_behavior")
    baccarat = safe_num(row, "baccarat_engagement_pct")
    blackjack = safe_num(row, "blackjack_engagement_pct")
    active_days = safe_num(row, "active_days")
    hours_trend = safe_num(row, "hours_played_w3_vs_w1")
    chase_trend = safe_num(row, "chase_rate_w3_vs_w1")
    ceiling = safe_num(row, "ceiling_pressure_score")
    range_width = safe_num(row, "range_width_score")
    stretch = safe_num(row, "stretch_capacity_score")
    rhythm = safe_num(row, "rhythm_score")
    session_fade = safe_num(row, "session_fade_score")
    post_loss = safe_num(row, "post_loss_response_score")
    game_affinity = safe_num(row, "game_affinity_score")
    table_fit = safe_num(row, "table_fit_score")
    hidden_opportunity = safe_num(row, "hidden_opportunity_score")
    confidence_need = safe_num(row, "confidence_need_score")
    primary_game = str(row.get("primary_game", "preferred game") or "preferred game").title()
    current_label = str(row.get("cohort_model_label", "") or "")
    target_label = str(row.get("target_better_cohort_label", "") or "")
    movement_context = f" Target cohort direction: {target_label}." if target_label else ""

    if tilt_prob >= 0.60 or post_loss >= 75 or chase >= 0.22 or volatility >= 80 or chase_trend > 0.08:
        confidence = max(tilt_prob, post_loss / 100.0, min(1.0, volatility / 100.0))
        return {
            "recommended_path": "Protect momentum after losses",
            "desired_behavior_change": "Stabilize play after difficult sequences.",
            "recommendation_action": "Protect momentum after losses",
            "recommendation_target": "Momentum protection path",
            "player_recommendation": "Protect the player's momentum before encouraging a bigger path.",
            "recommendation_reason": "Post-loss response, chase, or volatility is elevated versus the player's recent pattern." + movement_context,
            "supporting_signals": "post_loss_response_score, volatility_score_behavior, chase_rate",
            "success_metric": "Lower chase rate and healthier session continuation on the next trip.",
            "recommendation_confidence": confidence,
        }

    if bool(row.get("in_cohort_optimization_flag", False)):
        confidence = max(
            0.45,
            safe_num(row, "in_cohort_optimization_score"),
            hidden_opportunity / 100.0,
            rhythm / 100.0,
            table_fit / 100.0,
        )
        cohort_context = f" Current cohort: {current_label}." if current_label else ""
        return {
            "recommended_path": "Strengthen current cohort position",
            "desired_behavior_change": "Move the player closer to the center of their current behavioral cohort.",
            "recommendation_action": "Strengthen current cohort position",
            "recommendation_target": current_label or "Current cohort",
            "player_recommendation": "Optimize the player's current pattern before forcing a tier or cohort move.",
            "recommendation_reason": "The player shows value or opportunity signals but sits away from the center of their current cohort." + cohort_context,
            "supporting_signals": "cohort_distance_pct_in_cohort, in_cohort_opportunity_score, rhythm_score, table_fit_score, hidden_opportunity_score",
            "success_metric": "Lower cohort distance, stronger rhythm, and improved theo or worth while staying in the current cohort.",
            "recommendation_confidence": confidence,
        }

    if (limit_prob >= 0.55 or (stretch >= 70 and range_width >= 65)) and rhythm >= 45 and post_loss < 75:
        confidence = max(limit_prob, stretch / 100.0, range_width / 100.0)
        return {
            "recommended_path": "Invite to higher-limit path",
            "desired_behavior_change": "More play near the player's proven upper range.",
            "recommendation_action": "Invite to higher-limit path",
            "recommendation_target": "Higher-limit path",
            "player_recommendation": "Invite the player into a higher-limit path already supported by their behavior.",
            "recommendation_reason": "Wide range, stretch capacity, ceiling pressure, and rhythm indicate headroom beyond average play." + movement_context,
            "supporting_signals": "stretch_capacity_score, range_width_score, ceiling_pressure_score, rhythm_score",
            "success_metric": "Higher upper-range play share, higher next-trip max wager, or stronger rhythm.",
            "recommendation_confidence": confidence,
        }

    if baccarat_prob >= 0.58 or (baccarat >= 55 and baccarat >= blackjack + 15) or table_fit >= 75:
        confidence = max(baccarat_prob, table_fit / 100.0, game_affinity / 100.0)
        return {
            "recommended_path": "Improve table fit and access",
            "desired_behavior_change": "Shift more play into the environment where the player already engages best.",
            "recommendation_action": "Improve table fit and access",
            "recommendation_target": f"{primary_game} / preferred table-fit path",
            "player_recommendation": "Route the player toward the table/game environment their behavior already favors.",
            "recommendation_reason": "Game affinity, table-fit score, or Baccarat engagement is materially stronger than alternatives." + movement_context,
            "supporting_signals": "table_fit_score, game_affinity_score, primary_game, baccarat_engagement_pct",
            "success_metric": "Higher share of play in preferred game/table fit and stronger session continuation.",
            "recommendation_confidence": confidence,
        }

    if side_prob >= 0.60 or safe_num(row, "side_bet_rate") >= 0.25 or game_affinity >= 70:
        confidence = max(side_prob, game_affinity / 100.0, min(1.0, safe_num(row, "side_bet_rate") * 2.5))
        return {
            "recommended_path": "Expose to preferred game features",
            "desired_behavior_change": "Grow engagement through the game features the player already responds to.",
            "recommendation_action": "Expose to preferred game features",
            "recommendation_target": "Feature-fit path",
            "player_recommendation": "Use the player's side-bet or feature affinity as the growth path.",
            "recommendation_reason": "The player shows repeated feature or side-bet engagement." + movement_context,
            "supporting_signals": "side_bet_rate, side_handle_pct, game_affinity_score",
            "success_metric": "Higher feature-engagement share and stronger repeat-session behavior.",
            "recommendation_confidence": confidence,
        }

    if hidden_opportunity >= 70 and rhythm >= 45:
        confidence = max(hidden_opportunity / 100.0, engagement_prob)
        return {
            "recommended_path": "Develop hidden opportunity",
            "desired_behavior_change": "Convert strong behavior into more consistent high-value rhythm.",
            "recommendation_action": "Develop hidden opportunity",
            "recommendation_target": "Near-VIP growth path",
            "player_recommendation": "Treat the player as a behavioral growth candidate even if ADT understates them.",
            "recommendation_reason": "Low ADT context is offset by strong rhythm, range, or stretch behavior." + movement_context,
            "supporting_signals": "hidden_opportunity_score, rhythm_score, range_width_score, stretch_capacity_score",
            "success_metric": "More active days, stronger upper-range play share, and improved worth trajectory.",
            "recommendation_confidence": confidence,
        }

    if confidence_need >= 70 or session_fade >= 75:
        confidence = max(confidence_need / 100.0, session_fade / 100.0)
        return {
            "recommended_path": "Build confidence and engagement",
            "desired_behavior_change": "Help the player stay active without forcing escalation.",
            "recommendation_action": "Build confidence and engagement",
            "recommendation_target": "Confidence-building path",
            "player_recommendation": "Build engagement before pushing a higher-limit path.",
            "recommendation_reason": "The player has some capacity but shows fade, loss sensitivity, or uneven rhythm." + movement_context,
            "supporting_signals": "confidence_need_score, session_fade_score, stretch_capacity_score",
            "success_metric": "Longer controlled sessions and improved return rhythm.",
            "recommendation_confidence": confidence,
        }

    if engagement_prob >= 0.55 and (active_days < 7 or hours_trend > 0):
        return {
            "recommended_path": "Increase return rhythm",
            "desired_behavior_change": "Increase repeat trips or sessions.",
            "recommendation_action": "Increase return rhythm",
            "recommendation_target": "Return-rhythm path",
            "player_recommendation": "Nudge the player toward another trip/session.",
            "recommendation_reason": "The model sees behavior consistent with a near-term engagement lift." + movement_context,
            "supporting_signals": "pred_engagement_lift_prob, rhythm_score, hours_played trend",
            "success_metric": "More active days, more sessions, or more hours on the next observed period.",
            "recommendation_confidence": engagement_prob,
        }

    if bool(row.get("lowest_development_cohort_flag", False)):
        confidence = max(0.45, hidden_opportunity / 100.0, rhythm / 100.0)
        return {
            "recommended_path": "Develop from lowest cohort",
            "desired_behavior_change": "Move the player toward a stronger behavioral cohort.",
            "recommendation_action": "Develop from lowest cohort",
            "recommendation_target": "Entry development path",
            "player_recommendation": "Give this low-development cohort player a clear path toward stronger rhythm or fit.",
            "recommendation_reason": "The player is in one of the lowest-development cohorts." + movement_context,
            "supporting_signals": "cohort_development_rank, target_better_cohort_label, hidden_opportunity_score, rhythm_score",
            "success_metric": "Movement toward the target cohort through better rhythm, fit, or controlled range expansion.",
            "recommendation_confidence": confidence,
        }

    if bool(row.get("cohort_edge_to_better_flag", False)):
        confidence = max(0.45, safe_num(row, "cohort_edge_score"), hidden_opportunity / 100.0)
        return {
            "recommended_path": "Move toward adjacent better cohort",
            "desired_behavior_change": "Nudge the player across the behavioral boundary they are already near.",
            "recommendation_action": "Move toward adjacent better cohort",
            "recommendation_target": target_label or "Nearest higher-development cohort",
            "player_recommendation": "Focus on the smallest behavioral gap between the current cohort and the target cohort.",
            "recommendation_reason": "The player is near the edge of a higher-development cohort." + movement_context,
            "supporting_signals": "cohort_edge_score, target_better_cohort_margin, target_better_cohort_label",
            "success_metric": "Lower distance to the target cohort and eventual cohort movement.",
            "recommendation_confidence": confidence,
        }

    return {
        "recommended_path": "Maintain current trajectory",
        "desired_behavior_change": "Keep the player in their current positive pattern.",
        "recommendation_action": "Maintain current trajectory",
        "recommendation_target": "Current path",
        "player_recommendation": "Keep the player in the current experience and monitor movement.",
        "recommendation_reason": "No stronger behavioral action is currently more compelling than the player's existing pattern." + movement_context,
        "supporting_signals": "cohort_distance, rhythm_score, volatility_score_behavior",
        "success_metric": "Stable rhythm and no deterioration in volatility or engagement.",
        "recommendation_confidence": max(0.35, 1.0 - min(safe_num(row, "cohort_distance"), 1.0)),
    }


def infer_recommendations(cohort_path: Path, metadata_path: Path, output_dir: Path, recommend_all: bool = True) -> dict[str, object]:
    data = pd.read_csv(cohort_path, low_memory=False)
    total = data[data["period"].eq("Total")].copy()
    trends = build_trend_context(data)
    total = total.merge(trends, on="player_id", how="left")

    metadata = read_json(metadata_path)
    metadata["_metadata_dir"] = str(metadata_path.parent)
    scored = load_predictions(total, metadata)
    scored = add_recommendation_candidate_flags(scored)
    model_candidate = bool_series(scored, "priority_recommendation_candidate")
    scored["model_candidate_reason"] = scored["recommendation_candidate_reason"]
    if recommend_all:
        scored["recommendation_candidate"] = True
        scored.loc[
            scored["recommendation_candidate_reason"].eq("No recommendation candidate"),
            "recommendation_candidate_reason",
        ] = "Daily full-population recommendation inference"
    else:
        scored["recommendation_candidate"] = model_candidate
    recs = scored.apply(choose_recommendation, axis=1, result_type="expand")
    out = pd.concat([scored.reset_index(drop=True), recs.reset_index(drop=True)], axis=1)
    out["path_fit_score"] = numeric_series(out, "recommendation_confidence")

    keep_first = [
        "player_id",
        "latest_session_id",
        "cohort_id",
        "cohort_model_label",
        "target_better_cohort_id",
        "target_better_cohort_label",
        "target_better_cohort_margin",
        "cohort_edge_to_better_flag",
        "cohort_edge_score",
        "cohort_development_rank",
        "cohort_development_score",
        "recommendation_candidate",
        "recommendation_candidate_reason",
        "priority_recommendation_candidate",
        "model_candidate_reason",
        "in_cohort_optimization_flag",
        "in_cohort_optimization_score",
        "in_cohort_opportunity_score",
        "cohort_distance_pct_in_cohort",
        "cohort_distance",
        "cohort_confidence_score",
        "recommended_path",
        "desired_behavior_change",
        "player_recommendation",
        "recommendation_action",
        "recommendation_target",
        "recommendation_reason",
        "supporting_signals",
        "success_metric",
        "path_fit_score",
        "recommendation_confidence",
        "pred_engagement_lift_prob",
        "pred_tilt_risk_prob",
        "pred_baccarat_engagement_prob",
        "pred_side_bet_engagement_prob",
        "pred_limit_path_readiness_prob",
        "ceiling_pressure_score",
        "range_width_score",
        "stretch_capacity_score",
        "rhythm_score",
        "session_fade_score",
        "post_loss_response_score",
        "game_affinity_score",
        "table_fit_score",
        "hidden_opportunity_score",
        "confidence_need_score",
    ]
    cols = [c for c in keep_first if c in out.columns] + [c for c in out.columns if c not in keep_first]
    out = out[cols].sort_values(["path_fit_score", "player_id"], ascending=[False, True])

    path = output_dir / "player_recommendations.csv"
    safe_write_csv(out, path)
    candidates = out[out["recommendation_candidate"].fillna(False).astype(bool)].copy()
    candidate_path = output_dir / "player_recommendation_candidates.csv"
    safe_write_csv(candidates, candidate_path)
    write_json(
        {
            "rows": int(len(out)),
            "candidate_rows": int(len(candidates)),
            "recommend_all": bool(recommend_all),
            "output": str(path),
            "candidate_output": str(candidate_path),
            "score_fields": {
                "path_fit_score": "Primary path fit/signal-strength score.",
                "recommendation_confidence": "Legacy alias for path_fit_score; not an outcome probability.",
            },
        },
        output_dir / "recommendation_inference_metadata.json",
    )
    return {
        "rows": int(len(out)),
        "candidate_rows": int(len(candidates)),
        "recommend_all": bool(recommend_all),
        "output": str(path),
        "candidate_output": str(candidate_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce one recommendation per player.")
    parser.add_argument("--cohorts", type=Path, default=DEFAULT_OUTPUT_DIR / "player_cohorts.csv")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_MODEL_DIR / "recommendation_model_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--edge-candidates-only",
        action="store_true",
        help="Only generate recommendation text for cohort-edge or lowest-development candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(infer_recommendations(args.cohorts, args.metadata, args.output_dir, recommend_all=not args.edge_candidates_only))


if __name__ == "__main__":
    main()
