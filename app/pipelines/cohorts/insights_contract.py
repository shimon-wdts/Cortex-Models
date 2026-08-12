#!/usr/bin/env python3
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from app.inference.recommendation_contract import recommendation_deduplication_id


SPEC_VERSION = 1.0
DEFAULT_SOURCE = "cortex.models.cohort-recommendation"
SOURCE_BY_MODEL = {
    "PlayerPerformance": "cortex.models.player-performance",
    "PlayerCohort": "cortex.models.player-cohorts",
    "CohortTierLift": "cortex.models.cohort-tier-lift",
}
ENV = "prod"
MODEL_VERSION = "10.2.0"
APPLICATIONS = ["cortexFloor"]
PLAYER_SCORE_WEIGHTS = {
    "worth": 0.35,
    "deal_hold": 0.35,
    "frequency": 0.20,
    "volatility": 0.10,
}

BEHAVIOR_TRAITS = [
    {
        "key": "chases_after_losses",
        "category": "loss_response",
        "label": "Chases after losses",
        "score_field": "trait_chases_after_losses_score",
        "fallback_field": "chase_rate",
        "evidence": "Player increases wager after losing outcomes more often than peers.",
        "source_fields": ["chase_rate", "bet_increase_after_loss_flag"],
    },
    {
        "key": "stops_quickly_after_losses",
        "category": "loss_response",
        "label": "Stops quickly after losses",
        "score_field": "trait_stops_quickly_after_losses_score",
        "fallback_field": "post_loss_stop_rate",
        "evidence": "Player ends or pauses play after losing outcomes more often than peers.",
        "source_fields": ["post_loss_stop_rate", "loss_exit_rate"],
    },
    {
        "key": "stabilizes_after_losses",
        "category": "loss_response",
        "label": "Stabilizes after losses",
        "score_field": "trait_stabilizes_after_losses_score",
        "fallback_field": None,
        "evidence": "Player keeps wagering steadier after losses with lower chase and volatility signals.",
        "source_fields": ["chase_rate", "post_loss_stop_rate", "volatility_score_behavior"],
    },
    {
        "key": "regular_side_bet_player",
        "category": "side_bet_engagement",
        "label": "Regular side-bet player",
        "score_field": "trait_regular_side_bet_player_score",
        "fallback_field": "side_bet_rate",
        "evidence": "Player's side-bet share is elevated versus peers.",
        "source_fields": ["side_bet_rate", "side_handle_pct"],
    },
    {
        "key": "avoids_side_bets",
        "category": "side_bet_engagement",
        "label": "Avoids side bets",
        "score_field": "trait_avoids_side_bets_score",
        "fallback_field": None,
        "evidence": "Player's side-bet participation is low compared with peers.",
        "source_fields": ["side_bet_rate", "side_handle_pct"],
    },
    {
        "key": "expands_bet_range",
        "category": "bet_range",
        "label": "Expands bet range",
        "score_field": "trait_expands_bet_range_score",
        "fallback_field": "stretch_capacity_score",
        "evidence": "Player uses a wider wager range and plays near the upper range more often.",
        "source_fields": ["range_width_score", "stretch_capacity_score", "ceiling_pressure_score"],
    },
    {
        "key": "keeps_bet_size_stable",
        "category": "bet_range",
        "label": "Keeps bet size stable",
        "score_field": "trait_keeps_bet_size_stable_score",
        "fallback_field": None,
        "evidence": "Player's wager spread and volatility are comparatively low.",
        "source_fields": ["volatility_score_behavior", "range_width_score"],
    },
    {
        "key": "returns_frequently",
        "category": "return_rhythm",
        "label": "Returns frequently",
        "score_field": "trait_returns_frequently_score",
        "fallback_field": "rhythm_score",
        "evidence": "Player has a stronger active-day and session-return rhythm than peers.",
        "source_fields": ["active_days", "num_sessions", "session_continuation_rate", "rhythm_score"],
    },
    {
        "key": "infrequent_return_pattern",
        "category": "return_rhythm",
        "label": "Infrequent return pattern",
        "score_field": "trait_infrequent_return_pattern_score",
        "fallback_field": None,
        "evidence": "Player's return rhythm is lighter than peers.",
        "source_fields": ["active_days", "num_sessions", "rhythm_score"],
    },
    {
        "key": "fades_late_in_session",
        "category": "session_fade",
        "label": "Fades late in session",
        "score_field": "trait_fades_late_in_session_score",
        "fallback_field": "session_fade_score",
        "evidence": "Player shows higher late-session or post-loss fade signals.",
        "source_fields": ["session_fade_score", "post_loss_stop_rate", "loss_exit_rate"],
    },
    {
        "key": "sustains_longer_sessions",
        "category": "session_endurance",
        "label": "Sustains longer sessions",
        "score_field": "trait_sustains_longer_sessions_score",
        "fallback_field": "hours_played",
        "evidence": "Player sustains longer session duration and total hours versus peers.",
        "source_fields": ["avg_session_hours", "max_session_hours", "hours_played"],
    },
    {
        "key": "strong_game_preference",
        "category": "game_preference",
        "label": "Strong game preference",
        "score_field": "trait_strong_game_preference_score",
        "fallback_field": "game_concentration_pct",
        "evidence": "Player's play is concentrated in a preferred game type.",
        "source_fields": ["primary_game", "primary_game_pct", "game_concentration_pct"],
    },
    {
        "key": "game_flexible_player",
        "category": "game_preference",
        "label": "Game-flexible player",
        "score_field": "trait_game_flexible_player_score",
        "fallback_field": None,
        "evidence": "Player's play is spread across game types rather than concentrated.",
        "source_fields": ["game_concentration_pct", "baccarat_engagement_pct", "blackjack_engagement_pct"],
    },
]
BEHAVIOR_TRAIT_BY_KEY = {str(item["key"]): item for item in BEHAVIOR_TRAITS}
PRIMARY_RISK_TRAITS = {"chases_after_losses", "expands_bet_range", "fades_late_in_session"}
PRIMARY_BEHAVIOR_PRIORITY_GROUPS = [
    (
        70.0,
        {
            "chases_after_losses",
            "expands_bet_range",
            "fades_late_in_session",
        },
    ),
    (
        65.0,
        {
            "stops_quickly_after_losses",
            "stabilizes_after_losses",
            "returns_frequently",
            "infrequent_return_pattern",
            "sustains_longer_sessions",
            "game_flexible_player",
            "avoids_side_bets",
            "regular_side_bet_player",
        },
    ),
    (
        60.0,
        {
            "strong_game_preference",
            "keeps_bet_size_stable",
        },
    ),
]

ENTITY_FIELDS = [
    ("TABLE", ("table_id", "tableId", "table_code", "table"), True),
    ("Player", ("player_id", "playerId"), True),
    ("bet_id", ("bet_id", "betId"), False),
    ("game_id", ("game_id", "gameId"), False),
    ("topology_id", ("topology_id", "topologyId", "topologu_id"), False),
]


def safe_str(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return default if text.lower() == "nan" else text


def has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return safe_str(value) != ""


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return default
    return float(number)


def optional_float(value: Any, digits: int = 2) -> float | None:
    number = safe_float(value, None)
    return None if number is None else round(number, digits)


def optional_str(value: Any) -> str | None:
    text = safe_str(value)
    return text or None


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return safe_str(value).lower() in {"true", "1", "yes", "y"}


def score_0_100(value: Any, default: float | None = 0.0) -> float | None:
    raw = safe_float(value, default)
    if raw is None:
        return None
    if raw <= 1.0:
        raw *= 100.0
    return round(max(0.0, min(100.0, raw)), 1)


def score_1_100(value: Any, default: float | None = 1.0) -> float | None:
    raw = safe_float(value, default)
    if raw is None:
        return None
    if 0.0 <= raw < 1.0:
        raw *= 100.0
    return round(max(1.0, min(100.0, raw)), 1)


def score_probability(value: Any, default: float | None = 0.0) -> float | None:
    raw = safe_float(value, default)
    if raw is None:
        return None
    if raw > 1.0:
        raw /= 100.0
    return round(max(0.0, min(1.0, raw)), 4)


def confidence_from_score(score: float | None) -> str:
    if score is None:
        return "low"
    if score >= 75:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def severity_from_score(score: float | None, actionable: bool = True) -> str:
    if not actionable:
        return "info"
    if score is None:
        return "low"
    if score >= 85:
        return "critical"
    if score >= 70:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z")


def gaming_day() -> str:
    return pd.Timestamp.utcnow().date().isoformat()


def ttl_iso(days: int = 30) -> str:
    return (pd.Timestamp.utcnow() + pd.Timedelta(days=days)).isoformat().replace("+00:00", "Z")


def insight_entities(row: pd.Series) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entity_type, fields, present in ENTITY_FIELDS:
        entity_id = ""
        for field in fields:
            entity_id = safe_str(row.get(field))
            if entity_id:
                break
        if not entity_id or (entity_type, entity_id) in seen:
            continue
        seen.add((entity_type, entity_id))
        entities.append({"type": entity_type, "id": entity_id, "present_in_user_interface": present})
    return entities


def base_payload(
    *,
    model_type: str,
    severity: str,
    entities: list[dict[str, Any]],
    result: dict[str, Any],
    presentation: dict[str, Any],
    source: str | None = None,
    actions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "insights_id": f"evt_{uuid.uuid4().hex}",
        "occurred_at": now_iso(),
        "gaming_day": gaming_day(),
        "source": source or SOURCE_BY_MODEL.get(model_type, DEFAULT_SOURCE),
        "env": ENV,
        "version": SPEC_VERSION,
        "model": {"type": model_type, "version": MODEL_VERSION},
        "entity": entities,
        "severity": severity,
        "application": APPLICATIONS,
        "payload": {
            "result": result,
            "presentation": presentation,
            "actions": actions
            or {
                "available": ["approve", "decline", "export"],
                "default": "archive",
                "requires_reason_on": ["decline"],
                "export": {"formats": ["pdf", "csv"], "scope": "recommendation"},
            },
        },
    }


def first_score_0_100(*values: Any) -> float | None:
    for value in values:
        if not has_value(value):
            continue
        score = score_0_100(value, None)
        if score is not None:
            return score
    return None


def compute_player_score(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    components = player_score_components(row)
    if not components:
        return score_1_100(row.get("player_score"), None)
    weight_total = sum(PLAYER_SCORE_WEIGHTS.values()) or 1.0
    weighted = sum(components[key] * PLAYER_SCORE_WEIGHTS[key] for key in PLAYER_SCORE_WEIGHTS)
    return score_1_100(weighted / weight_total, None)


def player_score_components(row: pd.Series | None) -> dict[str, float]:
    if row is None:
        return {}
    return {
        "worth": score_1_100(row.get("worth_score"), 1.0) or 1.0,
        "deal_hold": score_1_100(row.get("deal_hold_score"), 1.0) or 1.0,
        "frequency": score_1_100(row.get("frequency_score"), 1.0) or 1.0,
        "volatility": score_1_100(row.get("volatility_score_behavior", row.get("volatility_score")), 1.0) or 1.0,
    }


def behavior_characteristics(row: pd.Series | None) -> dict[str, float]:
    if row is None:
        return {}

    characteristics: dict[str, float] = {}
    for trait in BEHAVIOR_TRAITS:
        key = str(trait["key"])
        score = behavior_trait_score(row, key)
        if score is not None:
            characteristics[key] = score
    return characteristics


def score_components(row: pd.Series | None) -> dict[str, float]:
    return player_score_components(row)


def behavior_trait_score(row: pd.Series, key: str) -> float | None:
    trait = BEHAVIOR_TRAIT_BY_KEY.get(key)
    if not trait:
        return None

    score = score_0_100(row.get(str(trait["score_field"])), None)
    if score is not None:
        return score

    if key == "stabilizes_after_losses":
        chase = score_0_100(row.get("chase_rate"), 0.0) or 0.0
        stop = score_0_100(row.get("post_loss_stop_rate"), 0.0) or 0.0
        volatility = score_0_100(row.get("volatility_score_behavior"), 0.0) or 0.0
        return round(max(0.0, min(100.0, 100.0 - (0.45 * chase + 0.30 * stop + 0.25 * volatility))), 1)
    if key == "regular_side_bet_player":
        side_rate = score_0_100(row.get("side_bet_rate"), 0.0) or 0.0
        side_handle = score_0_100(row.get("side_handle_pct"), 0.0) or 0.0
        return round(max(0.0, min(100.0, 0.70 * side_rate + 0.30 * side_handle)), 1)
    if key == "avoids_side_bets":
        regular = behavior_trait_score(row, "regular_side_bet_player") or 0.0
        return round(max(0.0, min(100.0, 100.0 - regular)), 1)
    if key == "expands_bet_range":
        range_width = score_0_100(row.get("range_width_score"), 0.0) or 0.0
        stretch = score_0_100(row.get("stretch_capacity_score"), 0.0) or 0.0
        ceiling = score_0_100(row.get("ceiling_pressure_score"), 0.0) or 0.0
        return round(max(0.0, min(100.0, 0.45 * range_width + 0.35 * stretch + 0.20 * ceiling)), 1)
    if key == "keeps_bet_size_stable":
        volatility = score_0_100(row.get("volatility_score_behavior"), 0.0) or 0.0
        range_width = score_0_100(row.get("range_width_score"), 0.0) or 0.0
        return round(max(0.0, min(100.0, 100.0 - (0.55 * volatility + 0.45 * range_width))), 1)
    if key == "infrequent_return_pattern":
        rhythm = score_0_100(row.get("rhythm_score"), 0.0) or 0.0
        return round(max(0.0, min(100.0, 100.0 - rhythm)), 1)
    if key == "sustains_longer_sessions":
        avg_session = score_0_100(row.get("avg_session_hours"), 0.0) or 0.0
        max_session = score_0_100(row.get("max_session_hours"), 0.0) or 0.0
        hours = score_0_100(row.get("hours_played"), 0.0) or 0.0
        return round(max(0.0, min(100.0, 0.45 * avg_session + 0.35 * hours + 0.20 * max_session)), 1)
    if key == "strong_game_preference":
        primary_game_pct = score_0_100(row.get("primary_game_pct"), 0.0) or 0.0
        concentration = score_0_100(row.get("game_concentration_pct"), 0.0) or 0.0
        return round(max(0.0, min(100.0, 0.65 * primary_game_pct + 0.35 * concentration)), 1)
    if key == "game_flexible_player":
        concentration = score_0_100(row.get("game_concentration_pct"), 0.0) or 0.0
        return round(max(0.0, min(100.0, 100.0 - concentration)), 1)

    fallback_field = trait.get("fallback_field")
    if fallback_field:
        return score_0_100(row.get(str(fallback_field)), None)
    return None


def behavior_trait_label(row: pd.Series, key: str) -> str:
    trait = BEHAVIOR_TRAIT_BY_KEY.get(key)
    if not trait:
        return key.replace("_", " ").title()
    if key == "strong_game_preference":
        primary_game = safe_str(row.get("primary_game")).upper()
        if primary_game and primary_game != "UNKNOWN":
            return f"Strong {primary_game.title()} preference"
    return str(trait["label"])


def behavior_item(
    row: pd.Series,
    key: str | None,
    score: float | None = None,
    label: str | None = None,
) -> dict[str, Any] | None:
    if not key:
        return None
    trait = BEHAVIOR_TRAIT_BY_KEY.get(key, {})
    item_label = label or behavior_trait_label(row, key)
    return {
        "key": key,
        "label": item_label,
        "score": score,
        "category": trait.get("category"),
        "characteristic": item_label,
        "evidence": trait.get("evidence"),
        "source_fields": trait.get("source_fields", []),
    }


def secondary_behavior_items(
    row: pd.Series,
    characteristics: dict[str, float],
    primary_key: str | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(
        ((key, score) for key, score in characteristics.items() if key != primary_key),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        behavior_item(row, key, score)
        for key, score in ranked[:limit]
        if behavior_item(row, key, score) is not None
    ]


def primary_behavior_key(row: pd.Series, characteristics: dict[str, float]) -> str | None:
    for threshold, group_keys in PRIMARY_BEHAVIOR_PRIORITY_GROUPS:
        candidates = [
            (key, score)
            for key, score in characteristics.items()
            if key in group_keys and isinstance(score, (int, float)) and score >= threshold
        ]
        if candidates:
            return max(candidates, key=lambda item: item[1])[0]

    scored = [(key, score) for key, score in characteristics.items() if isinstance(score, (int, float))]
    if scored:
        strongest_key, strongest_score = max(scored, key=lambda item: item[1])
        if strongest_score >= 60:
            return strongest_key

    trait_key = safe_str(row.get("primary_behavior_trait_key"))
    if trait_key in characteristics:
        return trait_key

    label = safe_str(row.get("primary_behavior")).lower()
    label_map = [
        ("chase", "chases_after_losses"),
        ("loss", "chases_after_losses"),
        ("side", "regular_side_bet_player"),
        ("volatile", "expands_bet_range"),
        ("volatility", "expands_bet_range"),
        ("engaged", "returns_frequently"),
        ("balanced", "stabilizes_after_losses"),
    ]
    for needle, key in label_map:
        if needle in label and key in characteristics:
            return key
    return None


def betting_style(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {}
    primary_game = safe_str(row.get("primary_game"), "Unknown")
    side_bet_intensity = safe_str(row.get("side_bet_intensity"), "Unknown")
    risk_label = safe_str(row.get("risk_volatility_label"), "Unknown")
    limit_readiness = first_score_0_100(row.get("pred_limit_path_readiness_prob"), row.get("stretch_capacity_score"))

    label_parts = []
    if primary_game and primary_game != "Unknown":
        label_parts.append(primary_game.title())
    if side_bet_intensity and side_bet_intensity != "Unknown":
        label_parts.append(f"{side_bet_intensity} side-bet")
    if risk_label and risk_label != "Unknown":
        label_parts.append(risk_label)

    return {
        "label": " / ".join(label_parts) if label_parts else "Unknown betting style",
        "primary_game": primary_game,
        "avg_bet": optional_float(row.get("avg_bet"), 2),
        "side_bet_intensity": side_bet_intensity,
        "risk_volatility": risk_label,
        "limit_readiness_score": limit_readiness,
    }


def cohort_result(row: pd.Series) -> dict[str, Any]:
    return {
        "cohort_id": optional_str(row.get("cohort_id")),
        "label": safe_str(row.get("cohort_model_label"), safe_str(row.get("cohort_label"), "Unknown cohort")),
        "distance": optional_float(row.get("cohort_distance"), 6),
        "confidence_probability": score_probability(row.get("cohort_confidence_score"), None),
        "confidence_score": score_0_100(row.get("cohort_confidence_score"), None),
        "development_rank": optional_float(row.get("cohort_development_rank"), 0),
        "development_score": optional_float(row.get("cohort_development_score"), 2),
    }


def target_better_cohort_result(row: pd.Series) -> dict[str, Any]:
    return {
        "cohort_id": optional_str(row.get("target_better_cohort_id")),
        "label": optional_str(row.get("target_better_cohort_label")),
        "distance": optional_float(row.get("target_better_cohort_distance"), 6),
        "margin": optional_float(row.get("target_better_cohort_margin"), 6),
        "edge_to_better": safe_bool(row.get("cohort_edge_to_better_flag")),
        "edge_score": score_0_100(row.get("cohort_edge_score"), None),
        "development_rank": optional_float(row.get("target_better_cohort_development_rank"), 0),
        "development_score": optional_float(row.get("target_better_cohort_development_score"), 2),
    }


def predicted_probabilities(row: pd.Series) -> dict[str, float | None]:
    return {
        "engagement_lift": score_probability(row.get("pred_engagement_lift_prob"), None),
        "tilt_risk": score_probability(row.get("pred_tilt_risk_prob"), None),
        "baccarat_engagement": score_probability(row.get("pred_baccarat_engagement_prob"), None),
        "side_bet_engagement": score_probability(row.get("pred_side_bet_engagement_prob"), None),
        "limit_path_readiness": score_probability(row.get("pred_limit_path_readiness_prob"), None),
    }


def signal_list(value: Any) -> list[str]:
    return [part.strip() for part in safe_str(value).split(",") if part.strip()]


def shared_model_result(row: pd.Series) -> dict[str, Any]:
    player_score = compute_player_score(row)
    characteristics = behavior_characteristics(row)
    components = score_components(row)
    style = betting_style(row)
    primary_key = primary_behavior_key(row, characteristics)
    primary = behavior_item(
        row,
        primary_key,
        characteristics.get(primary_key) if primary_key else None,
    )
    secondary = secondary_behavior_items(row, characteristics, primary_key)

    return {
        "player": {
            "player_id": optional_str(row.get("player_id")),
            "latest_session_id": optional_str(row.get("latest_session_id")),
        },
        "player_score": player_score,
        "score_components": components,
        "behavior_profile": {
            "primary_behavior": primary,
            "secondary_behaviors": secondary,
            "characteristics": characteristics,
        },
        "betting_style": style,
    }


def recommendation_result(row: pd.Series, action_type: str) -> dict[str, Any]:
    return {
        "recommended_path": optional_str(row.get("recommended_path")),
        "action_type": action_type,
        "action": optional_str(row.get("recommendation_action")),
        "target": optional_str(row.get("recommendation_target")),
        "text": optional_str(row.get("player_recommendation")),
        "reason": optional_str(row.get("recommendation_reason")),
        "supporting_signals": signal_list(row.get("supporting_signals")),
        "success_metric": optional_str(row.get("success_metric")),
        "path_fit_score": score_0_100(row.get("path_fit_score"), None),
        "recommendation_confidence": score_probability(row.get("recommendation_confidence"), None),
        "candidate": safe_bool(row.get("recommendation_candidate")),
        "candidate_reason": optional_str(row.get("recommendation_candidate_reason")),
    }


def recommendation_thresholds(metric: str, value: float) -> list[dict[str, Any]]:
    return [{"metric": metric, "operator": ">=", "value": value}]


def build_cohort_insight(row: pd.Series) -> dict[str, Any]:
    player_id = safe_str(row.get("player_id"), "unknown")
    cohort_label = safe_str(row.get("cohort_model_label"), safe_str(row.get("cohort_label"), "Unknown cohort"))
    target_label = safe_str(row.get("target_better_cohort_label"), "No higher-fit cohort identified")
    confidence_probability = score_probability(row.get("cohort_confidence_score"))
    confidence_score = score_0_100(row.get("cohort_confidence_score")) or 0.0
    edge_score = score_0_100(row.get("cohort_edge_score")) or 0.0
    confidence = confidence_from_score(confidence_score)
    shared = shared_model_result(row)
    rationale = (
        f"Player {player_id} is assigned to {cohort_label}. "
        f"Cohort confidence is {confidence_score:.1f}/100 and edge-to-target score is {edge_score:.1f}/100. "
        f"Target cohort for development tracking: {target_label}."
    )
    recommendation = {
        "id": 1,
        "action": {
            "type": "review_cohort_assignment",
            "cohort": cohort_label,
        },
        "text": f"Use {cohort_label} as the current behavioral cohort for Player {player_id}.",
        "rationale": rationale,
        "modeled_impact": {"value": confidence_score, "unit": "/100 cohort confidence"},
        "roi": {"value": edge_score, "unit": "/100 target-cohort edge"},
        "time_to_action": {"unit": "Minutes", "value": 0},
        "confidence": confidence,
        "time_to_live": ttl_iso(),
        "thresholds": recommendation_thresholds("payload.result.score", 0.2),
        "deduplication_id": recommendation_deduplication_id(
            {
                "model_type": "PlayerCohort",
                "player_id": player_id,
                "action_type": "review_cohort_assignment",
                "cohort": cohort_label,
            }
        ),
    }
    result = {
        "decision_class": "cohort_assignment",
        "score": confidence_probability,
        "confidence": confidence,
        "modeled_impact": {"value": edge_score, "unit": "/100 target-cohort edge", "horizon_min": 0},
        "expected_deficit": None,
        "cohort": cohort_result(row),
        "target_better_cohort": target_better_cohort_result(row),
        "behavior_profile": shared["behavior_profile"],
        "player_score": shared["player_score"],
        "score_components": shared["score_components"],
        "betting_style": shared["betting_style"],
    }
    return base_payload(
        model_type="PlayerCohort",
        severity=severity_from_score(confidence_score, actionable=False),
        entities=insight_entities(row),
        result=result,
        presentation={
            "headline": "Player Cohort Assignment",
            "trigger_metric": "behavioral cohort confidence",
            "recommendations": [recommendation],
            "chart": {
                "type": "bar",
                "y_label": "Score",
                "x_labels": ["Cohort confidence", "Target edge"],
                "series": [{"name": "Cohort", "points": [confidence_score, edge_score]}],
            },
        },
    )


def _tier_headline(path: str) -> str:
    lower = path.lower()
    if "invite" in lower or "higher-limit" in lower or "baccarat" in lower:
        return "VIP Baccarat Re-Engagement"
    if "session" in lower or "engagement" in lower:
        return "Session Increase"
    return "Player Tier Lift Opportunity"


def _tier_lift_projection_weeks(visits_per_week: float) -> int:
    if visits_per_week >= 3.5:
        return 1
    if visits_per_week >= 2.5:
        return 2
    if visits_per_week >= 1.5:
        return 3
    return 4


def _tier_lift_chart(
    row: pd.Series,
    *,
    lift: float,
    unit: str,
    range95: list[float] | None = None,
) -> dict[str, Any]:
    active_days = safe_float(row.get("active_days"), None)
    frequency_source = "active_days_over_3_weeks"
    if active_days is None or active_days <= 0:
        active_days = safe_float(row.get("num_sessions"), None)
        frequency_source = "sessions_over_3_weeks"
    if active_days is None or active_days <= 0:
        active_days = 3.0
        frequency_source = "default_one_visit_per_week"

    visits_per_week = round(active_days / 3.0, 2)
    weeks_to_goal = _tier_lift_projection_weeks(visits_per_week)
    x_labels = ["Now", *[f"Week {week}" for week in range(1, weeks_to_goal + 1)]]
    points = [round(lift * step / weeks_to_goal, 2) for step in range(weeks_to_goal + 1)]
    chart = {
        "type": "line",
        "y_label": f"Projected lift ({unit})",
        "x_labels": x_labels,
        "series": [{"name": "Projected lift", "points": points}],
        "projection": {
            "method": "frequency_paced",
            "strategy": "conservative_timeline",
            "visits_per_week": visits_per_week,
            "frequency_source": frequency_source,
            "weeks_to_goal": weeks_to_goal,
            "model_horizon_weeks": 1,
        },
    }
    if range95 is not None:
        chart["final_range95"] = range95
    return chart


def build_tier_lift_insight(row: pd.Series) -> dict[str, Any]:
    player_id = safe_str(row.get("player_id"), "unknown")
    path = safe_str(row.get("recommended_path"), "Player development path")
    headline = _tier_headline(path)
    target_label = safe_str(row.get("target_better_cohort_label"), "target cohort")
    cohort_label = safe_str(row.get("cohort_model_label"), "current cohort")
    raw_path_fit = row.get("path_fit_score")
    if not has_value(raw_path_fit):
        raw_path_fit = row.get("recommendation_confidence")
    path_fit_probability = score_probability(raw_path_fit)
    path_fit = score_0_100(raw_path_fit) or 0.0
    engagement_lift = score_0_100(row.get("pred_engagement_lift_prob")) or 0.0
    raw_current_theo = row.get("total_session_theo")
    if not has_value(raw_current_theo):
        raw_current_theo = row.get("theo")
    current_theo = round(safe_float(raw_current_theo, 0.0) or 0.0, 2)
    theo_lift = round(safe_float(row.get("avg_theo_delta"), 0.0) or 0.0, 2)
    expected_theo = round(current_theo + theo_lift, 2)
    theo_ci_low = round(safe_float(row.get("avg_theo_delta_ci_low"), theo_lift) or 0.0, 2)
    theo_ci_high = round(safe_float(row.get("avg_theo_delta_ci_high"), theo_lift) or 0.0, 2)
    theo_impact = {
        "current_theo": current_theo,
        "expected_theo": expected_theo,
        "theo_lift": theo_lift,
        "range95": [theo_ci_low, theo_ci_high],
    }
    expected_deficit = optional_float(row.get("expected_deficit"), 2)
    impact_value = theo_lift if theo_lift else engagement_lift
    impact_unit = "EV" if theo_lift else "/100 engagement lift"
    baseline_text = "baseline/no-offer comparison is tracked against the historical no-action path"
    confidence = confidence_from_score(path_fit)
    decision_class = "reactivation_opportunity"

    if headline == "VIP Baccarat Re-Engagement":
        rec_text = (
            f"Invite Player {player_id} within the next 7 days to a Hi-limit Baccarat table like BA0054 or similar. "
            f"Modeled uplift: {engagement_lift:.1f}/100 engagement lift; expected theo value: {expected_theo:.2f}; "
            f"baseline/no-offer comparison: no invite; confidence: {confidence}."
        )
    elif headline == "Session Increase":
        rec_text = (
            f"Offer Player {player_id} a session-increase path within the next 7 days. "
            f"Option 1 should increase visit rhythm; option 2 should protect current play quality. "
            f"Modeled uplift: {engagement_lift:.1f}/100 engagement lift; expected theo value: {expected_theo:.2f}; "
            f"baseline/no-offer comparison: no session action; confidence: {confidence}."
        )
    else:
        rec_text = (
            f"Move Player {player_id} toward {target_label} within the next 7 days. "
            f"Modeled uplift: {engagement_lift:.1f}/100 engagement lift; expected theo value: {expected_theo:.2f}; "
            f"baseline/no-offer comparison: no recommendation; confidence: {confidence}."
        )

    rationale = (
        f"Why this surfaced: Player {player_id} is currently in {cohort_label} and the recommended path is {path}. "
        f"Path fit is {path_fit:.1f}/100, target cohort is {target_label}, and {baseline_text}."
    )
    primary_text = safe_str(row.get("player_recommendation"), rec_text)
    primary_rationale = safe_str(row.get("recommendation_reason"), rationale)
    recommendation_target = safe_str(row.get("recommendation_target"), path)
    action_source = safe_str(row.get("recommendation_action"), path).lower()
    shared = shared_model_result(row)
    if "offer" in action_source or "invite" in action_source or "baccarat" in action_source:
        primary_action_type = "targeted_offer"
    elif "loss" in action_source or "momentum" in action_source or "protect" in action_source:
        primary_action_type = "momentum_protection"
    elif "session" in action_source:
        primary_action_type = "session_growth"
    else:
        primary_action_type = "tier_lift_path"

    primary_action = {
        "type": primary_action_type,
        "recommended_path": path,
        "target_cohort": target_label,
    }
    if headline == "VIP Baccarat Re-Engagement":
        primary_action.update(
            {
                "game_type": "baccarat",
                "segment": "higher_limit",
                "campaign_id": "reactivation-standard",
            }
        )

    recommendation = {
        "id": 1,
        "action": primary_action,
        "text": primary_text,
        "rationale": primary_rationale,
        "modeled_impact": {"value": impact_value, "unit": impact_unit, **theo_impact},
        "roi": {"value": path_fit, "unit": "/100 path fit"},
        "time_to_action": {"unit": "Minutes", "value": 30},
        "confidence": confidence,
        "time_to_live": ttl_iso(),
        "thresholds": recommendation_thresholds("payload.result.score", 0.8),
        "deduplication_id": recommendation_deduplication_id(
            {
                "model_type": "CohortTierLift",
                "player_id": player_id,
                "action_type": primary_action_type,
                "recommended_path": path,
                "target_cohort": target_label,
                "recommendation_target": recommendation_target,
            }
        ),
    }
    follow_up = {
        "id": 2,
        "action": {
            "type": "host_follow_up",
            "recommended_path": path,
            "target_cohort": target_label,
            "trigger": "primary_action_not_redeemed",
        },
        "depends_on": {
            "recommendation_id": 1,
            "condition": "not_redeemed",
        },
        "text": f"Assign host follow-up within 7 days if the {path.lower()} action is not redeemed.",
        "rationale": f"Keeps Player {player_id} aligned to {target_label} if the primary action does not convert.",
        "modeled_impact": {
            "value": round(impact_value * 0.85, 2),
            "unit": impact_unit,
            **theo_impact,
        },
        "roi": {"value": round(path_fit * 0.85, 1), "unit": "/100 path fit"},
        "time_to_action": {"unit": "Minutes", "value": 45},
        "confidence": confidence,
        "time_to_live": ttl_iso(),
        "thresholds": recommendation_thresholds("payload.result.score", 0.8),
        "deduplication_id": recommendation_deduplication_id(
            {
                "model_type": "CohortTierLift",
                "player_id": player_id,
                "action_type": "host_follow_up",
                "recommended_path": path,
                "target_cohort": target_label,
                "recommendation_target": recommendation_target,
                "trigger": "primary_action_not_redeemed",
            }
        ),
    }
    recommendations = [recommendation, follow_up]
    for recommendation_item in recommendations:
        recommendation_impact = recommendation_item["modeled_impact"]
        recommendation_lift = safe_float(recommendation_impact.get("theo_lift"), 0.0) or safe_float(
            recommendation_impact.get("value"), 0.0
        )
        recommendation_item["chart"] = _tier_lift_chart(
            row,
            lift=recommendation_lift,
            unit=safe_str(recommendation_impact.get("unit"), "lift"),
            range95=recommendation_impact.get("range95") if recommendation_impact.get("theo_lift") else None,
        )
    result = {
        "decision_class": decision_class,
        "score": path_fit_probability,
        "confidence": confidence,
        "modeled_impact": {"value": impact_value, "unit": impact_unit, "horizon_min": 10080},
        "expected_deficit": expected_deficit,
        "recommendation": recommendation_result(row, primary_action_type),
        "cohort": cohort_result(row),
        "target_better_cohort": target_better_cohort_result(row),
        "predicted_probabilities": predicted_probabilities(row),
        "behavior_profile": shared["behavior_profile"],
        "player_score": shared["player_score"],
        "betting_style": shared["betting_style"],
    }
    return base_payload(
        model_type="CohortTierLift",
        severity=severity_from_score(path_fit, actionable=True),
        entities=insight_entities(row),
        result=result,
        presentation={
            "headline": headline,
            "trigger_metric": "path fit and expected tier lift",
            "recommendations": recommendations,
        },
    )


def build_player_score_insight(row: pd.Series) -> dict[str, Any]:
    player_id = safe_str(row.get("player_id"), "unknown")
    score = compute_player_score(row) or 1.0
    score_probability_value = round(max(0.01, min(1.0, score / 100.0)), 4)
    components = player_score_components(row)
    worth = components["worth"]
    deal_hold = components["deal_hold"]
    frequency = components["frequency"]
    volatility = components["volatility"]
    confidence = confidence_from_score(score)
    shared = shared_model_result(row)
    rationale = (
        f"Why this surfaced: Player {player_id} has a composite score of {score:.1f}/100. "
        f"Components are worth {worth:.1f}, deal hold {deal_hold:.1f}, frequency {frequency:.1f}, and volatility {volatility:.1f}."
    )
    recommendation = {
        "id": 1,
        "action": {
            "type": "review_player_score",
        },
        "text": f"Review Player {player_id}'s score: {score:.1f}/100. Use it to prioritize host follow-up and cohort recommendations.",
        "rationale": rationale,
        "modeled_impact": {"value": score, "unit": "/100 player score"},
        "roi": {"value": worth, "unit": "/100 worth score"},
        "time_to_action": {"unit": "Days", "value": 7},
        "confidence": confidence,
        "time_to_live": ttl_iso(),
        "thresholds": recommendation_thresholds("payload.result.player_score", 55),
        "deduplication_id": recommendation_deduplication_id(
            {
                "model_type": "PlayerPerformance",
                "player_id": player_id,
                "action_type": "review_player_score",
                "player_score": f"{score:.1f}",
            }
        ),
    }
    result = {
        "decision_class": "player_score",
        "score": score_probability_value,
        "confidence": confidence,
        "modeled_impact": {"value": score, "unit": "/100 player score", "horizon_min": 10080},
        "expected_deficit": None,
        "player_score": score,
        "score_components": shared["score_components"],
        "cohort": cohort_result(row),
        "behavior_profile": shared["behavior_profile"],
        "betting_style": shared["betting_style"],
    }
    return base_payload(
        model_type="PlayerPerformance",
        severity=severity_from_score(score, actionable=True),
        entities=insight_entities(row),
        result=result,
        presentation={
            "headline": "Player Score",
            "trigger_metric": "composite player value score",
            "recommendations": [recommendation],
            "chart": {
                "type": "bar",
                "y_label": "Score",
                "x_labels": ["Worth", "Deal hold", "Frequency", "Volatility"],
                "series": [{"name": "Player score", "points": [worth, deal_hold, frequency, volatility]}],
            },
        },
    )


def merge_path_lift(recommendations: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    validation_path = output_dir / "validation" / "expected_lift_by_recommendation_path.csv"
    if not validation_path.exists() or "recommended_path" not in recommendations.columns:
        return recommendations
    path_lift = pd.read_csv(validation_path, low_memory=False)
    if "recommended_path" not in path_lift.columns:
        return recommendations
    keep = [
        "recommended_path",
        "avg_theo_delta",
        "avg_theo_delta_ci_low",
        "avg_theo_delta_ci_high",
        "engagement_lift_rate",
        "cohort_moved_up_rate",
        "cohort_centered_rate",
    ]
    keep = [col for col in keep if col in path_lift.columns]
    return recommendations.merge(path_lift[keep], on="recommended_path", how="left")


def total_period_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "period" not in df.columns:
        return df
    total = df[df["period"].astype(str).str.lower().eq("total")].copy()
    return total if not total.empty else df


def read_rows(output_dir: Path, preferred_name: str, fallback_name: str) -> pd.DataFrame | None:
    preferred = output_dir / preferred_name
    fallback = output_dir / fallback_name
    if preferred.exists():
        return total_period_rows(pd.read_csv(preferred, low_memory=False))
    if fallback.exists():
        return total_period_rows(pd.read_csv(fallback, low_memory=False))
    return None


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def player_json_path(output_dir: Path, folder_name: str, row: pd.Series) -> Path:
    player_id = safe_str(row.get("player_id"), "unknown")
    safe_player_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in player_id)
    return output_dir / folder_name / f"player_{safe_player_id}.json"


def write_player_insight_files(
    *,
    output_dir: Path,
    folder_name: str,
    rows: pd.DataFrame,
    builder: Any,
) -> list[str]:
    folder = output_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for _, row in rows.iterrows():
        out_path = player_json_path(output_dir, folder_name, row)
        write_json(out_path, builder(row))
        written.append(str(out_path))
    return written


def write_canonical_insight_outputs(output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}

    cohorts = read_rows(output_dir, "player_cohorts.csv", "cohort_inference_output.csv")
    if cohorts is not None:
        written = write_player_insight_files(
            output_dir=output_dir,
            folder_name="cohort_insights_output",
            rows=cohorts,
            builder=build_cohort_insight,
        )
        result["cohort_insights_output"] = written
        result["cohort_insight_rows"] = len(written)

    recs = read_rows(output_dir, "player_recommendations.csv", "recommendation_inference_output.csv")
    if recs is not None:
        recs = merge_path_lift(recs, output_dir)
        written = write_player_insight_files(
            output_dir=output_dir,
            folder_name="cohort_tier_lift_insights_output",
            rows=recs,
            builder=build_tier_lift_insight,
        )
        result["cohort_tier_lift_insights_output"] = written
        result["cohort_tier_lift_insight_rows"] = len(written)

    score_path = output_dir / "player_total_features.csv"
    if score_path.exists():
        scores = total_period_rows(pd.read_csv(score_path, low_memory=False))
        written = write_player_insight_files(
            output_dir=output_dir,
            folder_name="player_score_insights_output",
            rows=scores,
            builder=build_player_score_insight,
        )
        result["player_score_insights_output"] = written
        result["player_score_insight_rows"] = len(written)

    return result
