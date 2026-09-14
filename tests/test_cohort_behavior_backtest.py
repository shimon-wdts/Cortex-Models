from __future__ import annotations

import pandas as pd

from app.pipelines.cohorts.behavior_backtest import complete_months, run_behavior_backtest


def bet_row(player: str, day: int, wager: float, theo: float, session: str) -> dict[str, object]:
    start = pd.Timestamp(2025, 10, day, 10, tz="UTC")
    return {
        "BetId": f"{player}{day}",
        "SessionId": session,
        "PlayerId": player,
        "TableId": "1",
        "GamingDay": f"2025-10-{day:02d}",
        "Wager": wager,
        "TheoWin": theo,
        "CasinoWin": 0.0,
        "BetType": "BANKER",
        "TypeOfBet": "MAIN_BET",
        "PayoutCompleteDtm": start,
        "SessionStartDtm": start,
        "SessionEndDtm": start + pd.Timedelta(hours=1),
        "GameType_game": "BACCARAT",
    }


def test_awt_growth_and_behavior_adoption() -> None:
    rows = []
    for player in ["1", "2"]:
        rows.extend(
            [
                bet_row(player, 1, 100, 10, f"{player}-a"),
                bet_row(player, 8, 100, 10, f"{player}-b"),
                bet_row(player, 15, 100, 10, f"{player}-c"),
            ]
        )
    rows.extend(
        [
            bet_row("1", 22, 100, 10, "1-d"),
            bet_row("1", 23, 100, 10, "1-e"),
            bet_row("1", 24, 100, 10, "1-f"),
            bet_row("1", 28, 100, 10, "1-g"),
            bet_row("2", 28, 100, 10, "2-d"),
        ]
    )
    data = pd.DataFrame(rows)

    prepared_months = complete_months(
        pd.DataFrame({"gaming_day": pd.date_range("2025-10-01", "2025-10-28")})
    )
    assert prepared_months == ["2025-10"]

    summary, outcomes, metadata = run_behavior_backtest(
        data,
        months=["2025-10"],
        min_baseline_bets=1,
    )
    player_one = outcomes[outcomes["player_id"].eq("1")].iloc[0]
    assert player_one["baseline_awt"] == 10.0
    assert player_one["future_awt"] == 40.0
    assert player_one["awt_growth_pct"] == 300.0
    assert bool(player_one["followed_increase_return_rhythm"])
    assert metadata["primary_outcome"] == "AWT = TheoWin / observed weeks"
    rhythm = summary[summary["behavior_change"].eq("Increase return rhythm")].iloc[0]
    assert rhythm["followers"] == 1
