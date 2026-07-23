from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.pipelines.cohorts.build_features import (
    BET_USECOLS,
    GAME_USECOLS,
    SESSION_USECOLS,
    add_bet_sequence_features,
    build_player_period_features,
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
) -> CohortsFeatureResult:
    sessions = prepare_sessions_df(raw_sessions)
    games = prepare_games_df(raw_games)
    bets = prepare_bets_df(raw_bets, games, sessions)
    if bets.empty or sessions.empty:
        raise ValueError("Cohorts input produced no usable Week 1-3 bet/session rows.")

    bets = add_bet_sequence_features(bets)
    features, worth_ratio = build_player_period_features(sessions, bets)
    total = features[features["period"].eq("Total")].copy()
    metadata = {
        "raw_session_rows": int(len(raw_sessions)),
        "raw_bet_rows": int(len(raw_bets)),
        "raw_game_rows": int(len(raw_games)),
        "player_period_rows": int(len(features)),
        "player_total_rows": int(len(total)),
        "winner_loser_ratio": worth_ratio,
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


def prepare_sessions_df(sess: pd.DataFrame) -> pd.DataFrame:
    sess = normalize_columns(sess, SESSION_USECOLS)
    sess["PlayerId"] = clean_id(sess["PlayerId"])
    sess["SessionId"] = clean_id(sess["SessionId"])
    sess["TableId"] = clean_id(sess["TableId"])
    for col in ["SessionStartDtm", "SessionEndDtm", "FirstWagerGameStartDtm"]:
        sess[col] = pd.to_datetime(sess[col], utc=True, errors="coerce")
    sess["GamingDay"] = pd.to_datetime(sess["GamingDay"], errors="coerce").dt.normalize()
    sess["period"] = assign_period(sess["GamingDay"])
    sess = sess[sess["period"].ne("")].copy()
    for col in ["NumBets", "NumGamesElapsed", "NumGamesWithWager", "Turnover", "TheoWin", "PlayerWin", "Buyin"]:
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


def prepare_bets_df(bet: pd.DataFrame, games: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    bet = normalize_columns(bet, BET_USECOLS)
    bet["PlayerId"] = clean_id(bet["PlayerId"])
    bet["SessionId"] = clean_id(bet["SessionId"])
    bet["TableId"] = clean_id(bet["TableId"])
    bet["BetId"] = pd.to_numeric(bet["BetId"], errors="coerce")
    bet["GameId"] = pd.to_numeric(bet["GameId"], errors="coerce")
    bet["Wager"] = pd.to_numeric(bet["Wager"], errors="coerce").fillna(0.0)
    bet["CasinoWin"] = pd.to_numeric(bet["CasinoWin"], errors="coerce").fillna(0.0)
    bet["PayoutCompleteDtm"] = pd.to_datetime(bet["PayoutCompleteDtm"], utc=True, errors="coerce")
    bet["BetType"] = bet["BetType"].fillna("").astype(str).str.upper()
    bet["TypeOfBet"] = bet["TypeOfBet"].fillna("").astype(str).str.upper()
    bet["Status"] = bet["Status"].fillna("").astype(str).str.upper()

    game_small = games[["GameId", "GameStartDtm", "GamingDay", "GameType", "Outcome", "NumPlayers", "NumPositions"]]
    betg = bet.merge(game_small, on="GameId", how="left", validate="many_to_one")
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
    betg["GamingDay"] = pd.to_datetime(betg["GamingDay"], errors="coerce").fillna(betg["SessionGamingDay"])
    betg["GamingDay"] = pd.to_datetime(betg["GamingDay"], errors="coerce").dt.normalize()
    betg["period"] = assign_period(betg["GamingDay"])
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
    betg["TheoWin_row"] = pd.to_numeric(betg["SessionTheoWin"], errors="coerce").fillna(0.0) * share.fillna(0.0)
    return betg
