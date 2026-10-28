from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from app.pipelines.cohorts.build_features import (
    BET_USECOLS,
    GAME_USECOLS,
    ProgressCallback,
    SESSION_USECOLS,
    add_bet_sequence_features,
    build_player_period_aggregates,
    build_player_period_features,
    report_feature_progress,
    score_player_period_features,
)
from app.pipelines.cohorts.utils import assign_period, clean_id


@dataclass
class CohortsFeatureResult:
    player_period_features: pd.DataFrame
    player_total_features: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.player_total_features)


def build_features_from_frames(
    raw_sessions: pd.DataFrame,
    raw_bets: pd.DataFrame,
    raw_games: pd.DataFrame,
    observation_start: Any | None = None,
    progress: ProgressCallback | None = None,
    winner_loser_ratio: float | None = None,
    score_features: bool = True,
) -> CohortsFeatureResult:
    started = perf_counter()
    report_feature_progress(progress, "prepare_sessions", "started", rows=len(raw_sessions))
    sessions = prepare_sessions_df(raw_sessions, observation_start=observation_start)
    report_feature_progress(progress, "prepare_sessions", "completed", started_at=started, rows=len(sessions))

    started = perf_counter()
    report_feature_progress(progress, "prepare_games", "started", rows=len(raw_games))
    games = prepare_games_df(raw_games)
    report_feature_progress(progress, "prepare_games", "completed", started_at=started, rows=len(games))

    bets = prepare_bets_df(
        raw_bets,
        games,
        sessions,
        observation_start=observation_start,
        progress=progress,
    )
    if bets.empty or sessions.empty:
        raise ValueError("Cohorts input produced no usable Week 1-3 bet/session rows.")

    bets = add_bet_sequence_features(bets, progress=progress)
    if score_features:
        features, worth_ratio = build_player_period_features(
            sessions,
            bets,
            progress=progress,
            winner_loser_ratio=winner_loser_ratio,
        )
    else:
        features, worth_ratio = build_player_period_aggregates(
            sessions,
            bets,
            progress=progress,
            winner_loser_ratio=winner_loser_ratio,
        )
    total = features[features["period"].eq("Total")].copy()
    metadata = {
        "raw_session_rows": int(len(raw_sessions)),
        "raw_bet_rows": int(len(raw_bets)),
        "raw_game_rows": int(len(raw_games)),
        "player_period_rows": int(len(features)),
        "player_total_rows": int(len(total)),
        "winner_loser_ratio": worth_ratio,
        "observation_start": str(observation_start) if observation_start is not None else None,
    }
    return CohortsFeatureResult(
        player_period_features=features,
        player_total_features=total,
        metadata=metadata,
    )


def normalize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    lower_map = {c.lower(): c for c in out.columns}
    rename = {}
    for expected in cols:
        found = lower_map.get(expected.lower())
        if found is not None:
            rename[found] = expected
    out = out.rename(columns=rename)
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols]


def prepare_sessions_df(sess: pd.DataFrame, observation_start: Any | None = None) -> pd.DataFrame:
    sess = normalize_columns(sess, SESSION_USECOLS)
    sess["PlayerId"] = clean_id(sess["PlayerId"])
    sess["SessionId"] = clean_id(sess["SessionId"])
    sess["TableId"] = clean_id(sess["TableId"])
    for col in ["SessionStartDtm", "SessionEndDtm"]:
        sess[col] = pd.to_datetime(sess[col], utc=True, errors="coerce")
    sess["GamingDay"] = pd.to_datetime(sess["GamingDay"], errors="coerce").dt.normalize()
    sess["period"] = assign_period(sess["GamingDay"], start_day=observation_start)
    sess = sess[sess["period"].ne("")].copy()
    for col in ["NumBets", "Turnover", "TheoWin", "PlayerWin", "Buyin"]:
        sess[col] = pd.to_numeric(sess[col], errors="coerce").fillna(0.0)
    duration = (sess["SessionEndDtm"] - sess["SessionStartDtm"]).dt.total_seconds().clip(lower=0) / 3600.0
    fallback = sess["NumBets"].fillna(0.0) * 45.0 / 3600.0
    sess["session_hours"] = np.maximum(duration.fillna(0.0), fallback.fillna(0.0))
    sess["player_result"] = pd.to_numeric(sess["PlayerWin"], errors="coerce").fillna(0.0)
    sess["session_loss_amt"] = np.where(sess["player_result"] < 0, -sess["player_result"], 0.0)
    sess["session_win_amt"] = np.where(sess["player_result"] > 0, sess["player_result"], 0.0)
    return sess.sort_values(["PlayerId", "GamingDay", "SessionStartDtm", "SessionId"], kind="mergesort")


def prepare_games_df(game: pd.DataFrame) -> pd.DataFrame:
    game = normalize_columns(game, GAME_USECOLS)
    game["GameId"] = pd.to_numeric(game["GameId"], errors="coerce")
    game["GameStartDtm"] = pd.to_datetime(game["GameStartDtm"], utc=True, errors="coerce")
    game["GamingDay"] = pd.to_datetime(game["GamingDay"], errors="coerce").dt.normalize()
    game["GameType"] = game["GameType"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    return game


def prepare_bets_df(
    bet: pd.DataFrame,
    games: pd.DataFrame,
    sessions: pd.DataFrame,
    observation_start: Any | None = None,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    started = perf_counter()
    report_feature_progress(progress, "normalize_bets", "started", rows=len(bet))
    embedded_game_context = all(
        column in bet.columns
        for column in ("GameStartDtm", "GamingDay", "GameType", "Outcome", "NumPlayers", "NumPositions")
    )
    bet = normalize_columns(bet, BET_USECOLS)
    bet["PlayerId"] = clean_id(bet["PlayerId"])
    bet["SessionId"] = clean_id(bet["SessionId"])
    bet["BetId"] = pd.to_numeric(bet["BetId"], errors="coerce")
    bet["GameId"] = pd.to_numeric(bet["GameId"], errors="coerce")
    bet["Wager"] = pd.to_numeric(bet["Wager"], errors="coerce").fillna(0.0)
    bet["CasinoWin"] = pd.to_numeric(bet["CasinoWin"], errors="coerce").fillna(0.0)
    bet["BetTheoWin"] = pd.to_numeric(bet["BetTheoWin"], errors="coerce")
    bet["PayoutCompleteDtm"] = pd.to_datetime(bet["PayoutCompleteDtm"], utc=True, errors="coerce")
    bet["GameStartDtm"] = pd.to_datetime(bet["GameStartDtm"], utc=True, errors="coerce")
    bet["GamingDay"] = pd.to_datetime(bet["GamingDay"], errors="coerce").dt.normalize()
    bet["GameType"] = bet["GameType"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    bet["BetType"] = bet["BetType"].fillna("").astype(str).str.upper()
    bet["TypeOfBet"] = bet["TypeOfBet"].fillna("").astype(str).str.upper()
    report_feature_progress(progress, "normalize_bets", "completed", started_at=started, rows=len(bet))

    started = perf_counter()
    report_feature_progress(progress, "merge_game_context", "started", rows=len(bet))
    if embedded_game_context:
        betg = bet
    else:
        game_small = games[
            ["GameId", "GameStartDtm", "GamingDay", "GameType", "Outcome", "NumPlayers", "NumPositions"]
        ]
        betg = bet.drop(columns=["GameStartDtm", "GamingDay", "GameType", "Outcome", "NumPlayers", "NumPositions"])
        betg = betg.merge(game_small, on="GameId", how="left", validate="many_to_one")
    report_feature_progress(progress, "merge_game_context", "completed", started_at=started, rows=len(betg))

    started = perf_counter()
    report_feature_progress(progress, "merge_session_context", "started", rows=len(betg))
    session_small = sessions[
        ["SessionId", "PlayerId", "GamingDay", "Turnover", "TheoWin", "PlayerWin", "period"]
    ].rename(
        columns={
            "GamingDay": "SessionGamingDay",
            "Turnover": "SessionTurnover",
            "TheoWin": "SessionTheoWin",
            "PlayerWin": "SessionPlayerWin",
            "period": "SessionPeriod",
        }
    )
    betg = betg.merge(session_small, on=["SessionId", "PlayerId"], how="left", validate="many_to_one")
    report_feature_progress(progress, "merge_session_context", "completed", started_at=started, rows=len(betg))

    started = perf_counter()
    report_feature_progress(progress, "enrich_bets", "started", rows=len(betg))
    betg["GamingDay"] = pd.to_datetime(betg["GamingDay"], errors="coerce").fillna(betg["SessionGamingDay"])
    betg["GamingDay"] = pd.to_datetime(betg["GamingDay"], errors="coerce").dt.normalize()
    betg["period"] = assign_period(betg["GamingDay"], start_day=observation_start)
    betg["period"] = betg["period"].where(betg["period"].ne(""), betg["SessionPeriod"])
    betg = betg[betg["period"].isin(["Week 1", "Week 2", "Week 3"])].copy()
    betg["event_time"] = betg["GameStartDtm"].fillna(betg["PayoutCompleteDtm"])
    betg["is_side_bet"] = (
        betg["TypeOfBet"].ne("MAIN_BET")
        | betg["BetType"].str.contains("PAIR|TIE|JACKPOT|SIDE|LUCKY", regex=True, na=False)
    ).astype(int)
    session_bets = betg.groupby("SessionId")["BetId"].transform("count").replace(0, np.nan)
    share = betg["Wager"] / pd.to_numeric(betg["SessionTurnover"], errors="coerce").replace(0, np.nan)
    share = share.fillna(1.0 / session_bets)
    session_allocated_theo = (
        pd.to_numeric(betg["SessionTheoWin"], errors="coerce").fillna(0.0) * share.fillna(0.0)
    )
    betg["TheoWin_row"] = betg["BetTheoWin"].where(betg["BetTheoWin"].notna(), session_allocated_theo)
    report_feature_progress(progress, "enrich_bets", "completed", started_at=started, rows=len(betg))
    return betg


def combine_feature_batches(
    batches: list[CohortsFeatureResult],
    *,
    winner_loser_ratio: float,
    progress: ProgressCallback | None = None,
) -> CohortsFeatureResult:
    """Combine small unscored aggregates and apply global population scoring once."""
    started = perf_counter()
    aggregate_rows = sum(len(batch.player_period_features) for batch in batches)
    report_feature_progress(progress, "combine_feature_batches", "started", rows=aggregate_rows)
    if not batches:
        empty = pd.DataFrame()
        report_feature_progress(progress, "combine_feature_batches", "completed", started_at=started, rows=0)
        return CohortsFeatureResult(empty, empty.copy(), {"winner_loser_ratio": winner_loser_ratio})

    aggregates = pd.concat(
        [batch.player_period_features for batch in batches],
        ignore_index=True,
        copy=False,
    )
    report_feature_progress(
        progress,
        "combine_feature_batches",
        "completed",
        started_at=started,
        rows=len(aggregates),
    )
    features = score_player_period_features(aggregates, progress=progress)
    total = features[features["period"].eq("Total")].copy()
    return CohortsFeatureResult(
        player_period_features=features,
        player_total_features=total,
        metadata={
            "player_period_rows": int(len(features)),
            "player_total_rows": int(len(total)),
            "winner_loser_ratio": winner_loser_ratio,
            "feature_batches": len(batches),
        },
    )
