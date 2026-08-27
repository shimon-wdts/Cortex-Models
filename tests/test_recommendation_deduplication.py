from __future__ import annotations

import re

import pandas as pd

from app.inference.recommendation_contract import recommendation_deduplication_id
from app.models.pipeline_contracts import ExecutionMode, RunContext
from app.pipelines.cohorts.insights_contract import (
    betting_style,
    build_cohort_insight,
    build_player_score_insight,
    build_tier_lift_insight,
)
from app.pipelines.lucky6_bigtiger.build_features import Lucky6FeatureRecord
from app.pipelines.lucky6_bigtiger.inference import _prediction_payload, advantageous_level
from app.pipelines.predicted_fills.inference import build_fill_alerts_json


def assert_sha_only(recommendation: dict[str, object]) -> None:
    assert "deduplication" not in recommendation
    assert re.fullmatch(r"[0-9a-f]{64}", str(recommendation["deduplication_id"]))


def test_recommendation_deduplication_id_is_canonical() -> None:
    fields = {
        "model_type": "ShoeAdvantage",
        "game_id": "116000235",
        "side_bet": "lucky6",
    }
    reordered = {
        "side_bet": "lucky6",
        "game_id": "116000235",
        "model_type": "ShoeAdvantage",
    }

    expected = "bce1e269819cd575abbd7afa1e4a45b1772deed33a065a88bd82c5750380c9fb"
    assert recommendation_deduplication_id(fields) == expected
    assert recommendation_deduplication_id(reordered) == expected


def test_shoe_advantage_uses_game_and_side_bet() -> None:
    record = Lucky6FeatureRecord(
        hand_id=47,
        shoe_id="1160004",
        game_id="116000235",
        event_ts=pd.Timestamp("2026-06-16T11:45:00.974Z"),
        game_start_ts=pd.Timestamp("2026-06-16T11:43:15.670Z"),
        gaming_day="2026-06-16",
        table_id="71",
        table_name="BA0071",
        pit_name="Baccarat",
        gaming_area="Main Floor",
        feature_row=None,
        cards_remaining=182,
        decks_remaining=3.5,
        history_size=30,
    )
    context = RunContext(
        model_name="ShoeAdvantage",
        run_id="test-run",
        execution_mode=ExecutionMode.MANUAL,
        parameters={"model_version": "recommended", "feature_version": "1.0"},
    )

    event = _prediction_payload(
        record,
        {"lucky6": 0.01209733, "big_tiger": -0.00416181},
        context,
        "recommended",
        "2026-07-28T11:49:09.896797+00:00",
    )
    recommendations = event["payload"]["presentation"]["recommendations"]

    assert len(recommendations) == 2
    assert [recommendation["action"]["side_bet"] for recommendation in recommendations] == [
        "LUCKY SIX",
        "BIG TIGER",
    ]
    assert event["payload"]["presentation"]["chart"]["x_labels"] == ["LUCKY SIX", "BIG TIGER"]
    for recommendation in recommendations:
        assert_sha_only(recommendation)
        action = recommendation["action"]
        side_bet = action["side_bet"]
        assert action["game_id"] == "116000235"
        assert action["shoe_id"] == "1160004"
        assert action["table_id"] == "71"
        assert recommendation["deduplication_id"] == recommendation_deduplication_id(
            {
                "model_type": "ShoeAdvantage",
                "game_id": "116000235",
                "side_bet": "lucky6" if side_bet == "LUCKY SIX" else "big_tiger",
            }
        )
    assert recommendations[0]["action"]["advantageous_level"] == "l"
    assert recommendations[1]["action"]["advantageous_level"] is None
    assert recommendations[0]["deduplication_id"] != recommendations[1]["deduplication_id"]


def test_shoe_advantage_level_boundaries() -> None:
    assert advantageous_level(-0.01) is None
    assert advantageous_level(0.0) is None
    assert advantageous_level(0.000001) == "l"
    assert advantageous_level(0.0244) == "l"
    assert advantageous_level(0.024401) == "m"
    assert advantageous_level(0.0588) == "m"
    assert advantageous_level(0.058801) == "h"


def test_predicted_fills_includes_severity_and_transfer_source() -> None:
    events = build_fill_alerts_json(
        pd.DataFrame(
            [
                {
                    "table_id": "174",
                    "snapshot_ts": "2026-07-26T03:45:00Z",
                    "need_prob": 0.91,
                    "need_pred": 1,
                    "decision_threshold": 0.60,
                    "route_v2_primary_dispatch_table_id": "175",
                }
            ]
        ),
        limit=1,
    )
    recommendations = events[0]["payload"]["presentation"]["recommendations"]
    cage_fill, table_transfer = recommendations

    assert_sha_only(cage_fill)
    assert cage_fill["deduplication_id"] == recommendation_deduplication_id(
        {
            "model_type": "PredictedFills",
            "table_id": "174",
            "action_type": "cage_fill",
            "severity": "critical",
        }
    )

    assert_sha_only(table_transfer)
    assert table_transfer["deduplication_id"] == recommendation_deduplication_id(
        {
            "model_type": "PredictedFills",
            "table_id": "174",
            "action_type": "table_transfer",
            "severity": "critical",
            "source_table_id": "175",
        }
    )


def test_player_cohort_uses_player_action_and_cohort() -> None:
    event = build_cohort_insight(
        pd.Series(
            {
                "player_id": "1000214",
                "cohort_model_label": "High Value",
                "target_better_cohort_label": "VIP",
                "cohort_confidence_score": 0.78,
                "cohort_edge_score": 0.64,
            }
        )
    )
    recommendation = event["payload"]["presentation"]["recommendations"][0]

    assert_sha_only(recommendation)
    assert recommendation["deduplication_id"] == recommendation_deduplication_id(
        {
            "model_type": "PlayerCohort",
            "player_id": "1000214",
            "action_type": "review_cohort_assignment",
            "cohort": "High Value",
        }
    )


def test_tier_lift_includes_specific_recommendation_target() -> None:
    event = build_tier_lift_insight(
        pd.Series(
            {
                "player_id": "1000214",
                "recommended_path": "Invite to higher-limit path",
                "recommendation_action": "Invite player to baccarat offer",
                "recommendation_target": "Higher-limit path",
                "target_better_cohort_label": "VIP",
                "cohort_model_label": "High Value",
                "path_fit_score": 0.88,
                "pred_engagement_lift_prob": 0.74,
                "total_session_theo": 1000,
                "active_days": 3,
                "last_gaming_day": "2026-07-21",
                "primary_game": "BACCARAT",
                "primary_game_pct": 100,
                "side_bet_intensity": "High",
            }
        )
    )
    primary, follow_up = event["payload"]["presentation"]["recommendations"]

    assert_sha_only(primary)
    assert primary["deduplication_id"] == recommendation_deduplication_id(
        {
            "model_type": "CohortTierLift",
            "player_id": "1000214",
            "action_type": "targeted_offer",
            "recommended_path": "Invite to higher-limit path",
            "target_cohort": "VIP",
            "recommendation_target": "Higher-limit path",
        }
    )
    assert primary["modeled_impact"]["current_theo"] == 333.33
    assert primary["modeled_impact"]["expected_theo"] == 575.59
    assert primary["modeled_impact"]["theo_lift"] == 242.26
    assert primary["modeled_impact"]["expected_theo_lift"] == 242.26
    assert primary["modeled_impact"]["theo_growth_pct"] == 72.6773
    assert primary["modeled_impact"]["predicted_growth_pct"] == 72.6773
    assert primary["modeled_impact"]["range95"] == [143.56, 363.51]
    assert primary["modeled_impact"]["prediction"] == {
        "method": "historical_behavior_frequency_recency_v1",
        "historical_behavior_change": "Invite to higher-limit path",
        "historical_median_awt_growth_pct": 94.2,
        "historical_median_adt_growth_pct": 17.1,
        "historical_players": 133,
        "visits_per_week": 1.0,
        "frequency_factor": 0.75,
        "recency_days": 0,
        "recency_factor": 1.0,
        "path_fit_factor": 0.97,
        "outcome_definition": "weekly Theo (AWT)",
        "historical_window": "Crowne 2025-10 through 2025-11",
        "range95_method": "scaled bootstrap CI95 for historical median AWT growth",
    }
    assert primary["chart"]["x_labels"] == ["Now", "Week 1", "Week 2", "Week 3", "Week 4"]
    assert primary["chart"]["series"] == [
        {"name": "Recommended", "points": [333.33, 393.89, 454.46, 515.02, 575.59]},
        {"name": "Baseline (No action)", "points": [333.33, 333.33, 333.33, 333.33, 333.33]},
    ]
    assert primary["chart"]["note"] == "Baseline assumes current Theo remains unchanged if no recommendation is taken."
    assert primary["chart"]["final_range95"] == [143.56, 363.51]
    assert primary["chart"]["projection"] == {
        "method": "frequency_paced",
        "strategy": "conservative_timeline",
        "baseline_method": "current_theo_flat",
        "visits_per_week": 1.0,
        "frequency_source": "active_days_over_3_weeks",
        "weeks_to_goal": 4,
        "model_horizon_weeks": 1,
    }
    assert primary["time_to_action"] == {"unit": "Days", "value": 14}
    assert primary["thresholds"] == [
        {
            "metric": "payload.result.modeled_impact.theo_growth_pct",
            "operator": ">=",
            "value": 5.0,
        }
    ]
    assert primary["rationale"] == (
        "This player strongly prefers Baccarat and frequently uses side bets, but visits about once per week. "
        "A carefully managed higher-limit option may better match their demonstrated play and support continued "
        "engagement."
    )
    assert event["payload"]["result"]["recommendation"]["reason"] == primary["rationale"]

    assert follow_up["modeled_impact"]["current_theo"] == 333.33
    assert follow_up["modeled_impact"]["expected_theo"] == 539.25
    assert follow_up["modeled_impact"]["theo_lift"] == 205.92
    assert follow_up["modeled_impact"]["theo_growth_pct"] == 61.78
    assert follow_up["modeled_impact"]["range95"] == [122.03, 308.98]
    assert follow_up["chart"]["series"] == [
        {"name": "Recommended", "points": [333.33, 384.81, 436.29, 487.77, 539.25]},
        {"name": "Baseline (No action)", "points": [333.33, 333.33, 333.33, 333.33, 333.33]},
    ]
    assert follow_up["chart"]["final_range95"] == [122.03, 308.98]
    assert follow_up["time_to_action"] == {"unit": "Days", "value": 21}
    assert follow_up["rationale"] == (
        "If the player does not respond to the initial recommendation, a host follow-up may improve engagement "
        "through a more personal interaction."
    )
    assert "chart" not in event["payload"]["presentation"]
    assert event["payload"]["presentation"]["subtitle"] == "This player shows strong potential to move toward VIP."
    assert event["payload"]["result"]["modeled_impact"]["theo_growth_pct"] == 72.6773

    assert_sha_only(follow_up)
    assert follow_up["deduplication_id"] == recommendation_deduplication_id(
        {
            "model_type": "CohortTierLift",
            "player_id": "1000214",
            "action_type": "host_follow_up",
            "recommended_path": "Invite to higher-limit path",
            "target_cohort": "VIP",
            "recommendation_target": "Higher-limit path",
            "trigger": "primary_action_not_redeemed",
        }
    )


def test_tier_lift_uses_latest_table_as_entity_and_retains_player_id() -> None:
    event = build_tier_lift_insight(
        pd.Series(
            {
                "player_id": "1000214",
                "latest_table_id": "71",
                "recommended_path": "Improve table fit and access",
                "path_fit_score": 0.8,
                "active_days": 3,
            }
        )
    )

    assert event["entity"] == [
        {"type": "TABLE", "id": "71", "present_in_user_interface": True}
    ]
    assert event["player_id"] == "1000214"
    assert event["table_id"] == "71"


def test_tier_lift_omits_entity_when_latest_table_is_unavailable() -> None:
    event = build_tier_lift_insight(
        pd.Series(
            {
                "player_id": "1000214",
                "recommended_path": "Maintain current trajectory",
                "path_fit_score": 0.8,
                "active_days": 3,
            }
        )
    )

    assert event["entity"] == []
    assert event["player_id"] == "1000214"
    assert "table_id" not in event


def test_tier_lift_projection_shortens_for_more_frequent_visits() -> None:
    expected_weeks = {3: 4, 6: 3, 9: 2, 12: 1}
    for active_days, weeks_to_goal in expected_weeks.items():
        event = build_tier_lift_insight(
            pd.Series(
                {
                    "player_id": f"player-{active_days}",
                    "path_fit_score": 0.8,
                    "pred_engagement_lift_prob": 0.7,
                    "total_session_theo": 1000,
                    "avg_theo_delta": 200,
                    "active_days": active_days,
                }
            )
        )
        recommendations = event["payload"]["presentation"]["recommendations"]
        for recommendation in recommendations:
            chart = recommendation["chart"]
            assert chart["projection"]["visits_per_week"] == active_days / 3
            assert chart["projection"]["weeks_to_goal"] == weeks_to_goal
            assert len(chart["x_labels"]) == weeks_to_goal + 1
            assert chart["series"][0]["points"][-1] == recommendation["modeled_impact"]["expected_theo"]
            assert chart["series"][1]["points"] == [333.33] * (weeks_to_goal + 1)


def test_tier_lift_time_to_action_uses_visit_frequency() -> None:
    expected_days = {
        1: (21, 21),
        3: (14, 21),
        9: (7, 14),
    }
    for active_days, (primary_days, follow_up_days) in expected_days.items():
        event = build_tier_lift_insight(
            pd.Series(
                {
                    "player_id": f"player-{active_days}",
                    "path_fit_score": 0.8,
                    "total_session_theo": 1000,
                    "avg_theo_delta": 200,
                    "active_days": active_days,
                }
            )
        )
        primary, follow_up = event["payload"]["presentation"]["recommendations"]
        assert primary["time_to_action"] == {"unit": "Days", "value": primary_days}
        assert follow_up["time_to_action"] == {"unit": "Days", "value": follow_up_days}


def test_tier_lift_exposes_five_percent_threshold_without_suppressing_output() -> None:
    event = build_tier_lift_insight(
        pd.Series(
            {
                "player_id": "zero-theo-player",
                "path_fit_score": 0.8,
                "total_session_theo": 0,
                "avg_theo_delta": 0,
                "active_days": 3,
            }
        )
    )
    primary = event["payload"]["presentation"]["recommendations"][0]
    assert primary["modeled_impact"]["theo_growth_pct"] is None
    assert primary["thresholds"] == [
        {
            "metric": "payload.result.modeled_impact.theo_growth_pct",
            "operator": ">=",
            "value": 5.0,
        }
    ]


def test_tier_lift_subtitle_handles_same_cohort() -> None:
    event = build_tier_lift_insight(
        pd.Series(
            {
                "player_id": "1000214",
                "cohort_model_label": "High Value",
                "target_better_cohort_label": "High Value",
                "path_fit_score": 0.8,
                "total_session_theo": 1000,
                "avg_theo_delta": 200,
                "active_days": 3,
            }
        )
    )
    assert event["payload"]["presentation"]["subtitle"] == (
        "This player shows strong potential to strengthen their position within High Value."
    )


def test_tier_lift_uses_product_rationale_for_every_path() -> None:
    expected_fragments = {
        "Protect momentum after losses": "more measured experience after losses",
        "Strengthen current cohort position": "more established in Current Cohort",
        "Invite to higher-limit path": "carefully managed higher-limit option",
        "Improve table fit and access": "familiar Baccarat table",
        "Expose to preferred game features": "Highlighting familiar game features",
        "Develop hidden opportunity": "gradual development plan",
        "Build confidence and engagement": "gradual, familiar experience",
        "Increase return rhythm": "timely invitation",
        "Develop from lowest cohort": "develop toward Target Cohort",
        "Move toward adjacent better cohort": "behaving similarly to Target Cohort",
        "Maintain current trajectory": "maintaining the current experience",
    }
    for path, expected_fragment in expected_fragments.items():
        event = build_tier_lift_insight(
            pd.Series(
                {
                    "player_id": "1000214",
                    "recommended_path": path,
                    "recommendation_action": path,
                    "cohort_model_label": "Current Cohort",
                    "target_better_cohort_label": "Target Cohort",
                    "path_fit_score": 0.8,
                    "primary_game": "BACCARAT",
                    "primary_game_pct": 100,
                    "side_bet_intensity": "High",
                    "active_days": 1,
                }
            )
        )
        primary, follow_up = event["payload"]["presentation"]["recommendations"]
        assert primary["rationale"].startswith(
            "This player strongly prefers Baccarat and frequently uses side bets, but visits infrequently."
        )
        assert expected_fragment in primary["rationale"]
        assert event["payload"]["result"]["recommendation"]["reason"] == primary["rationale"]
        assert follow_up["rationale"] == (
            "If the player does not respond to the initial recommendation, a host follow-up may improve engagement "
            "through a more personal interaction."
        )


def test_tier_lift_unknown_path_uses_safe_rationale_fallback() -> None:
    event = build_tier_lift_insight(
        pd.Series(
            {
                "player_id": "1000214",
                "recommended_path": "Future recommendation path",
                "path_fit_score": 0.8,
                "active_days": 3,
            }
        )
    )
    rationale = event["payload"]["presentation"]["recommendations"][0]["rationale"]
    assert rationale == (
        "This player visits about once per week. This recommendation follows the player's demonstrated preferences "
        "and is intended to improve engagement gradually."
    )


def test_player_performance_includes_displayed_score() -> None:
    event = build_player_score_insight(
        pd.Series(
            {
                "player_id": "1000214",
                "worth_score": 73.4,
                "deal_hold_score": 73.4,
                "frequency_score": 73.4,
                "volatility_score_behavior": 73.4,
            }
        )
    )
    recommendation = event["payload"]["presentation"]["recommendations"][0]

    assert event["payload"]["result"]["player_score"] == 73.4
    assert_sha_only(recommendation)
    assert recommendation["deduplication_id"] == recommendation_deduplication_id(
        {
            "model_type": "PlayerPerformance",
            "player_id": "1000214",
            "action_type": "review_player_score",
            "player_score": "73.4",
        }
    )


def test_betting_style_uses_product_taxonomy() -> None:
    expected_labels = {
        "Balanced": "Stable",
        "Engaged": "Progressive",
        "Side Bet Heavy": "Side-Bet Heavy",
        "High Volatility": "Volatile",
        "Chases Losses": "Risk Wagerer",
    }
    for primary_behavior, expected_label in expected_labels.items():
        style = betting_style(
            pd.Series(
                {
                    "primary_behavior": primary_behavior,
                    "primary_game": "BACCARAT",
                    "avg_bet": 125.5,
                    "side_bet_intensity": "Medium",
                    "risk_volatility_label": "Medium Risk",
                    "pred_limit_path_readiness_prob": 0.45,
                }
            )
        )
        assert style == {
            "label": expected_label,
            "primary_game": "BACCARAT",
            "avg_bet": 125.5,
            "side_bet_intensity": "Medium",
            "risk_volatility": "Medium Risk",
            "limit_readiness_score": 45.0,
        }
