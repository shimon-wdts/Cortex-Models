from __future__ import annotations

import re

import pandas as pd

from app.inference.recommendation_contract import recommendation_deduplication_id
from app.models.pipeline_contracts import ExecutionMode, RunContext
from app.pipelines.cohorts.insights_contract import (
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
    for recommendation in recommendations:
        assert_sha_only(recommendation)
        action = recommendation["action"]
        side_bet = action["side_bet"]
        assert action["game_id"] == "116000235"
        assert action["shoe_id"] == "1160004"
        assert recommendation["deduplication_id"] == recommendation_deduplication_id(
            {
                "model_type": "ShoeAdvantage",
                "game_id": "116000235",
                "side_bet": side_bet,
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
