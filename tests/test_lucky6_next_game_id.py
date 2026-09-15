from __future__ import annotations

import json

import pandas as pd

from app.pipelines.lucky6_bigtiger.build_features import build_feature_dataset
from app.pipelines.lucky6_bigtiger.fetch_data import normalize_t_game_rows


def _game(
    game_id: str,
    hand: int,
    start: str,
    payout: str = "",
    *,
    shoe_id: str = "shoe-1",
    gaming_day: str = "2026-09-09",
    table_id: str = "71",
) -> dict[str, str]:
    completed = bool(payout)
    return {
        "GameId": game_id,
        "ShoeId": shoe_id,
        "ShoeGameCount": str(hand),
        "CardP1": "A" if completed else "",
        "CardP2": "2" if completed else "",
        "CardP3": "",
        "CardB1": "3" if completed else "",
        "CardB2": "4" if completed else "",
        "CardB3": "",
        "PlayerScore": "3" if completed else "",
        "BankerScore": "7" if completed else "",
        "Outcome": "BANKER" if completed else "",
        "GameStartDtm": start,
        "PayoutCompleteDtm": payout,
        "GamingDay": gaming_day,
        "TableId": table_id,
        "TableName": "BA0071",
        "PitName": "Baccarat",
        "GamingArea": "Main Floor",
        "GameType": "BACCARAT",
        "GameStatus": "COMPLETED" if completed else "OPEN",
    }


def _build(rows: list[dict[str, str]], start: str, end: str):
    return build_feature_dataset(
        normalize_t_game_rows(pd.DataFrame(rows)),
        publish_start_ts=pd.Timestamp(start),
        publish_end_ts=pd.Timestamp(end),
    )


def test_prediction_waits_until_next_game_id_exists() -> None:
    result = _build(
        [_game("game-1", 1, "2026-09-09T10:00:00Z", "2026-09-09T10:01:00Z")],
        "2026-09-09T10:00:00Z",
        "2026-09-09T10:02:00Z",
    )

    assert result.records == []


def test_prediction_uses_open_next_games_id_and_completed_source_features() -> None:
    rows = [
        _game("game-1", 1, "2026-09-09T10:00:00Z", "2026-09-09T10:01:00Z"),
        _game("game-2", 2, "2026-09-09T10:02:00Z", "2026-09-09T10:03:00Z"),
        _game("game-3", 3, "2026-09-09T10:04:00Z"),
    ]

    result = _build(rows, "2026-09-09T10:00:00Z", "2026-09-09T10:05:00Z")

    assert [record.game_id for record in result.records] == ["game-2", "game-3"]
    assert [record.hand_id for record in result.records] == [2, 3]
    assert [record.history_size for record in result.records] == [1, 2]
    assert result.records[-1].feature_row is not None
    assert result.records[-1].game_start_ts == pd.Timestamp("2026-09-09T10:04:00Z")
    assert result.records[-1].event_ts == pd.Timestamp("2026-09-09T10:03:00Z")
    assert result.records[-1].source_row["GameId"] == "game-3"


def test_prediction_is_emitted_when_delayed_source_completion_makes_pair_ready() -> None:
    rows = [
        _game("game-1", 1, "2026-09-09T10:00:00Z", "2026-09-09T10:06:00Z"),
        _game("game-2", 2, "2026-09-09T10:02:00Z"),
    ]

    result = _build(rows, "2026-09-09T10:05:00Z", "2026-09-09T10:10:00Z")

    assert [record.game_id for record in result.records] == ["game-2"]


def test_nonconsecutive_game_is_not_mislabeled() -> None:
    rows = [
        _game("game-1", 1, "2026-09-09T10:00:00Z", "2026-09-09T10:01:00Z"),
        _game("game-3", 3, "2026-09-09T10:02:00Z"),
    ]

    result = _build(rows, "2026-09-09T10:00:00Z", "2026-09-09T10:05:00Z")

    assert result.records == []


def test_reused_shoe_id_is_isolated_by_gaming_day_and_table() -> None:
    rows = [
        _game(
            "old-game-1",
            1,
            "2026-09-08T10:00:00Z",
            "2026-09-08T10:01:00Z",
            gaming_day="2026-09-08",
        ),
        _game("old-game-2", 2, "2026-09-08T10:02:00Z", gaming_day="2026-09-08"),
        _game("table-71-game-1", 1, "2026-09-09T10:00:00Z", "2026-09-09T10:01:00Z"),
        _game("table-71-game-2", 2, "2026-09-09T10:02:00Z"),
        _game(
            "table-72-game-1",
            1,
            "2026-09-09T10:00:00Z",
            "2026-09-09T10:01:00Z",
            table_id="72",
        ),
        _game("table-72-game-2", 2, "2026-09-09T10:02:00Z", table_id="72"),
    ]

    result = _build(rows, "2026-09-09T00:00:00Z", "2026-09-09T23:59:59Z")

    assert [record.game_id for record in result.records] == ["table-71-game-2", "table-72-game-2"]
    assert [record.table_id for record in result.records] == ["71", "72"]
    assert [record.shoe_id for record in result.records] == ["shoe-1", "shoe-1"]
    assert [record.history_size for record in result.records] == [1, 1]


def test_diagnostics_show_selected_next_game_id() -> None:
    messages: list[str] = []
    rows = [
        _game("game-1", 1, "2026-09-09T10:00:00Z", "2026-09-09T10:01:00Z"),
        _game("game-2", 2, "2026-09-09T10:02:00Z"),
    ]

    build_feature_dataset(
        normalize_t_game_rows(pd.DataFrame(rows)),
        publish_start_ts=pd.Timestamp("2026-09-09T10:00:00Z"),
        publish_end_ts=pd.Timestamp("2026-09-09T10:05:00Z"),
        diagnostic_logger=messages.append,
    )

    events = [json.loads(message) for message in messages]
    matched = next(event for event in events if event["event"] == "shoe_advantage_next_game_matched")
    assert matched["source_game_id"] == "game-1"
    assert matched["expected_next_game_count"] == 2
    assert matched["candidate_game_id"] == "game-2"
    assert matched["selected_next_game_id"] == "game-2"


def test_diagnostics_show_when_next_game_is_missing() -> None:
    messages: list[str] = []

    build_feature_dataset(
        normalize_t_game_rows(
            pd.DataFrame([_game("game-1", 1, "2026-09-09T10:00:00Z", "2026-09-09T10:01:00Z")])
        ),
        publish_start_ts=pd.Timestamp("2026-09-09T10:00:00Z"),
        publish_end_ts=pd.Timestamp("2026-09-09T10:05:00Z"),
        diagnostic_logger=messages.append,
    )

    events = [json.loads(message) for message in messages]
    missing = next(event for event in events if event["event"] == "shoe_advantage_next_game_missing")
    assert missing["source_game_id"] == "game-1"
    assert missing["expected_next_game_count"] == 2
    assert missing["candidate_present"] is False
    assert missing["selected_next_game_id"] is None
