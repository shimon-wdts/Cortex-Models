from __future__ import annotations

import pandas as pd

from app.pipelines.cohorts.insights_contract import build_tier_lift_insight
from app.pipelines.cohorts.source import build_features_from_frames


def test_bet_theo_backfills_zero_session_theo() -> None:
    sessions = pd.DataFrame(
        [
            {
                "SessionId": 1,
                "TableId": 7,
                "PlayerId": 7295,
                "SessionStartDtm": "2026-07-01T10:00:00Z",
                "SessionEndDtm": "2026-07-01T10:30:00Z",
                "FirstWagerGameStartDtm": "2026-07-01T10:00:00Z",
                "GamingDay": "2026-07-01",
                "NumBets": 1,
                "NumGamesElapsed": 1,
                "NumGamesWithWager": 1,
                "Turnover": 0,
                "TheoWin": 0,
                "PlayerWin": 0,
                "Buyin": 0,
                "AdjustedTheoWin": 0,
                "GameType": "BACCARAT",
                "PitName": "PIT-1",
                "GamingArea": "MAIN",
                "TableName": "BA0007",
            }
        ]
    )
    games = pd.DataFrame(
        [
            {
                "GameId": 10,
                "GameStartDtm": "2026-07-01T10:01:00Z",
                "GamingDay": "2026-07-01",
                "GameType": "BACCARAT",
                "Outcome": "BANKER",
                "GameResult": "BANKER",
                "ShoeId": 20,
                "ShoeGameCount": 1,
                "TableId": 7,
                "TableName": "BA0007",
                "PitName": "PIT-1",
                "GamingArea": "MAIN",
                "NumPlayers": 1,
                "NumPositions": 7,
            }
        ]
    )
    bets = pd.DataFrame(
        [
            {
                "BetId": 100,
                "GameId": 10,
                "SessionId": 1,
                "PlayerId": 7295,
                "TableId": 7,
                "BetType": "BANKER",
                "TypeOfBet": "MAIN_BET",
                "ShortBetNameEn": "Banker",
                "Wager": 300,
                "CasinoWin": 10,
                "BetTheoWin": 8.5,
                "Status": "COMPLETE",
                "PayoutCompleteDtm": "2026-07-01T10:02:00Z",
            }
        ]
    )

    result = build_features_from_frames(sessions, bets, games)
    total = result.player_total_features.iloc[0]

    assert total["theo"] == 8.5
    assert total["total_session_theo"] == 8.5
    assert total["latest_table_id"] == "7"

    tier_row = total.copy()
    tier_row["path_fit_score"] = 0.9
    tier_row["recommended_path"] = "Improve table fit and access"
    tier_row["recommendation_action"] = "Improve table fit and access"
    tier_row["recommendation_target"] = "Baccarat / preferred table-fit path"
    tier_row["cohort_model_label"] = "Current cohort"
    tier_row["target_better_cohort_label"] = "Target cohort"
    event = build_tier_lift_insight(tier_row)

    assert event["entity"] == [
        {"type": "Player", "id": "7295", "present_in_user_interface": True}
    ]
    assert event["player_id"] == "7295"
    assert event["table_id"] == "7"
    impact = event["payload"]["result"]["modeled_impact"]
    assert impact["current_theo"] == 2.83
    assert impact["expected_theo"] == 3.0
    assert impact["theo_lift"] == 0.17
    assert impact["predicted_growth_pct"] == 5.8568
    for recommendation in event["payload"]["presentation"]["recommendations"]:
        assert recommendation["modeled_impact"]["current_theo"] == 2.83
        baseline = next(series for series in recommendation["chart"]["series"] if series["name"] == "Baseline (No action)")
        assert baseline["points"] == [2.83] * len(recommendation["chart"]["x_labels"])
