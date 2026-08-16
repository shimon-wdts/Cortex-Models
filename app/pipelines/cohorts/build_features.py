#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BET_FILE, DEFAULT_BASE_PATH, DEFAULT_OUTPUT_DIR, GAME_FILE, MIN_BETS, SESSION_FILE
from .utils import (
    add_total_period,
    assign_period,
    clean_id,
    ensure_dir,
    normalize_path,
    robust_pct_rank,
    safe_divide,
    safe_write_csv,
    write_json,
)


SESSION_USECOLS = [
    "SessionId",
    "TableId",
    "PlayerId",
    "SessionStartDtm",
    "SessionEndDtm",
    "FirstWagerGameStartDtm",
    "GamingDay",
    "NumBets",
    "NumGamesElapsed",
    "NumGamesWithWager",
    "Turnover",
    "TheoWin",
    "PlayerWin",
    "Buyin",
    "AdjustedTheoWin",
    "GameType",
    "PitName",
    "GamingArea",
    "TableName",
]

BET_USECOLS = [
    "BetId",
    "GameId",
    "SessionId",
    "PlayerId",
    "TableId",
    "BetType",
    "TypeOfBet",
    "ShortBetNameEn",
    "Wager",
    "CasinoWin",
    "BetTheoWin",
    "Status",
    "PayoutCompleteDtm",
]

GAME_USECOLS = [
    "GameId",
    "GameStartDtm",
    "GamingDay",
    "GameType",
    "Outcome",
    "GameResult",
    "ShoeId",
    "ShoeGameCount",
    "TableId",
    "TableName",
    "PitName",
    "GamingArea",
    "NumPlayers",
    "NumPositions",
]

BEHAVIOR_TRAIT_LABELS = {
    "chases_after_losses": "Chases after losses",
    "stops_quickly_after_losses": "Stops quickly after losses",
    "stabilizes_after_losses": "Stabilizes after losses",
    "regular_side_bet_player": "Regular side-bet player",
    "avoids_side_bets": "Avoids side bets",
    "expands_bet_range": "Expands bet range",
    "keeps_bet_size_stable": "Keeps bet size stable",
    "returns_frequently": "Returns frequently",
    "infrequent_return_pattern": "Infrequent return pattern",
    "fades_late_in_session": "Fades late in session",
    "sustains_longer_sessions": "Sustains longer sessions",
    "strong_game_preference": "Strong game preference",
    "game_flexible_player": "Game-flexible player",
}


def clamp_score_1_100(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(1.0).clip(lower=1.0, upper=100.0)


def read_existing_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in usecols if c in header.columns]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    for col in usecols:
        if col not in df.columns:
            df[col] = np.nan
    return df[usecols]


def prepare_sessions(path: Path) -> pd.DataFrame:
    sess = read_existing_csv(path, SESSION_USECOLS)
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
    sess = sess.sort_values(["PlayerId", "GamingDay", "SessionStartDtm", "SessionId"], kind="mergesort")
    return sess


def prepare_games(path: Path) -> pd.DataFrame:
    game = read_existing_csv(path, GAME_USECOLS)
    game["GameId"] = pd.to_numeric(game["GameId"], errors="coerce")
    game["GameStartDtm"] = pd.to_datetime(game["GameStartDtm"], utc=True, errors="coerce")
    game["GamingDay"] = pd.to_datetime(game["GamingDay"], errors="coerce").dt.normalize()
    game["GameType"] = game["GameType"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    return game


def prepare_bets(path: Path, games: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    bet = read_existing_csv(path, BET_USECOLS)
    bet["PlayerId"] = clean_id(bet["PlayerId"])
    bet["SessionId"] = clean_id(bet["SessionId"])
    bet["TableId"] = clean_id(bet["TableId"])
    bet["BetId"] = pd.to_numeric(bet["BetId"], errors="coerce")
    bet["GameId"] = pd.to_numeric(bet["GameId"], errors="coerce")
    bet["Wager"] = pd.to_numeric(bet["Wager"], errors="coerce").fillna(0.0)
    bet["CasinoWin"] = pd.to_numeric(bet["CasinoWin"], errors="coerce").fillna(0.0)
    bet["BetTheoWin"] = pd.to_numeric(bet["BetTheoWin"], errors="coerce")
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
    share = safe_divide(betg["Wager"], pd.to_numeric(betg["SessionTurnover"], errors="coerce"))
    share = share.fillna(1.0 / session_bets)
    session_allocated_theo = (
        pd.to_numeric(betg["SessionTheoWin"], errors="coerce").fillna(0.0) * share.fillna(0.0)
    )
    betg["TheoWin_row"] = betg["BetTheoWin"].where(betg["BetTheoWin"].notna(), session_allocated_theo)
    return betg


def add_bet_sequence_features(betg: pd.DataFrame) -> pd.DataFrame:
    data = betg.sort_values(["PlayerId", "SessionId", "event_time", "BetId"], kind="mergesort").copy()
    data["prev_wager"] = data.groupby(["PlayerId", "SessionId"])["Wager"].shift(1)
    data["prev_casino_win"] = data.groupby(["PlayerId", "SessionId"])["CasinoWin"].shift(1)
    data["prev_is_loss"] = data["prev_casino_win"].gt(0).astype(int)
    data["bet_increase_after_loss_flag"] = (
        data["prev_is_loss"].eq(1) & data["Wager"].gt(data["prev_wager"])
    ).astype(int)
    data["post_loss_stop_flag"] = (
        data.groupby(["PlayerId", "SessionId"])["prev_is_loss"].shift(-1).isna() & data["CasinoWin"].gt(0)
    ).astype(int)
    return data


def build_player_period_features(sessions: pd.DataFrame, betg: pd.DataFrame) -> pd.DataFrame:
    player_actual = betg.groupby("PlayerId", dropna=False)["CasinoWin"].sum()
    winners = int((player_actual < 0).sum())
    losers = int((player_actual > 0).sum())
    winner_loser_ratio = winners / losers if losers else 0.25
    betg["worth_row"] = np.maximum.reduce(
        [
            betg["TheoWin_row"].fillna(0.0).to_numpy(),
            (betg["CasinoWin"].fillna(0.0) * winner_loser_ratio).to_numpy(),
            np.zeros(len(betg)),
        ]
    )
    all_bets = add_total_period(betg, "period")
    all_sessions = add_total_period(sessions, "period")
    all_bets["side_wager"] = np.where(all_bets["is_side_bet"].eq(1), all_bets["Wager"], 0.0)
    period_max_wager = all_bets.groupby(["period", "PlayerId"], dropna=False)["Wager"].transform("max")
    all_bets["near_ceiling_flag"] = (all_bets["Wager"].ge(period_max_wager * 0.80) & period_max_wager.gt(0)).astype(int)

    bet_agg = (
        all_bets.groupby(["period", "PlayerId"], dropna=False)
        .agg(
            bet_count=("BetId", "count"),
            turnover=("Wager", "sum"),
            avg_bet=("Wager", "mean"),
            std_bet=("Wager", "std"),
            min_bet=("Wager", "min"),
            max_bet=("Wager", "max"),
            theo=("TheoWin_row", "sum"),
            casino_win=("CasinoWin", "sum"),
            worth_raw=("worth_row", "sum"),
            chase_count=("bet_increase_after_loss_flag", "sum"),
            chase_rate=("bet_increase_after_loss_flag", "mean"),
            post_loss_stop_rate=("post_loss_stop_flag", "mean"),
            side_bet_count=("is_side_bet", "sum"),
            side_handle=("side_wager", "sum"),
            ceiling_pressure_rate=("near_ceiling_flag", "mean"),
        )
        .reset_index()
    )
    bet_agg["side_bet_rate"] = safe_divide(bet_agg["side_bet_count"], bet_agg["bet_count"]).fillna(0.0)
    bet_agg["side_handle_pct"] = safe_divide(bet_agg["side_handle"], bet_agg["turnover"]).fillna(0.0) * 100.0
    bet_agg["bet_spread_ratio"] = safe_divide(bet_agg["max_bet"] - bet_agg["min_bet"], bet_agg["avg_bet"]).fillna(0.0)

    game_agg = (
        all_bets.groupby(["period", "PlayerId", "GameType"], dropna=False)
        .agg(game_turnover=("Wager", "sum"), game_bets=("BetId", "count"))
        .reset_index()
    )
    total_game = game_agg.groupby(["period", "PlayerId"], dropna=False)["game_turnover"].transform("sum")
    game_agg["game_pct"] = safe_divide(game_agg["game_turnover"], total_game).fillna(0.0) * 100.0
    primary = (
        game_agg.sort_values(["period", "PlayerId", "game_turnover"], ascending=[True, True, False])
        .drop_duplicates(["period", "PlayerId"])
        .rename(columns={"GameType": "primary_game", "game_pct": "primary_game_pct"})
    )[["period", "PlayerId", "primary_game", "primary_game_pct"]]
    game_pivot = game_agg.pivot_table(
        index=["period", "PlayerId"], columns="GameType", values="game_pct", aggfunc="sum", fill_value=0.0
    ).reset_index()
    for col in ["BACCARAT", "BLACKJACK"]:
        if col not in game_pivot.columns:
            game_pivot[col] = 0.0
    game_pivot = game_pivot.rename(
        columns={"BACCARAT": "baccarat_engagement_pct", "BLACKJACK": "blackjack_engagement_pct"}
    )
    game_pivot["game_concentration_pct"] = game_pivot[["baccarat_engagement_pct", "blackjack_engagement_pct"]].max(axis=1)

    sess_agg = (
        all_sessions.groupby(["period", "PlayerId"], dropna=False)
        .agg(
            num_sessions=("SessionId", "nunique"),
            latest_session_id=("SessionId", "last"),
            active_days=("GamingDay", "nunique"),
            hours_played=("session_hours", "sum"),
            avg_session_hours=("session_hours", "mean"),
            max_session_hours=("session_hours", "max"),
            total_session_turnover=("Turnover", "sum"),
            total_session_theo=("TheoWin", "sum"),
            total_player_win=("PlayerWin", "sum"),
            cash_buy_in=("Buyin", "sum"),
            avg_buyin=("Buyin", "mean"),
            max_session_loss=("session_loss_amt", "max"),
            avg_session_loss=("session_loss_amt", "mean"),
            max_session_win=("session_win_amt", "max"),
            last_gaming_day=("GamingDay", "max"),
        )
        .reset_index()
    )
    sess_agg["session_continuation_rate"] = safe_divide(sess_agg["num_sessions"], sess_agg["active_days"]).fillna(0.0)
    sess_agg["loss_exit_rate"] = safe_divide(sess_agg["max_session_loss"], sess_agg["turnover"] if "turnover" in sess_agg else sess_agg["total_session_turnover"]).fillna(0.0)
    sess_agg["credit_line"] = sess_agg["cash_buy_in"]

    features = bet_agg.merge(sess_agg, on=["period", "PlayerId"], how="outer")
    session_theo = pd.to_numeric(features["total_session_theo"], errors="coerce").fillna(0.0)
    bet_theo = pd.to_numeric(features["theo"], errors="coerce").fillna(0.0)
    features["total_session_theo"] = session_theo.where(session_theo.ne(0.0), bet_theo)
    features = features.merge(primary, on=["period", "PlayerId"], how="left")
    features = features.merge(
        game_pivot[["period", "PlayerId", "baccarat_engagement_pct", "blackjack_engagement_pct", "game_concentration_pct"]],
        on=["period", "PlayerId"],
        how="left",
    )
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["PlayerId"] = features["PlayerId"].astype(str)
    features["eligible"] = features["bet_count"].ge(MIN_BETS)
    features["worth_score"] = clamp_score_1_100(features.groupby("period")["worth_raw"].transform(robust_pct_rank))
    features["frequency_score"] = clamp_score_1_100(features.groupby("period")["active_days"].transform(robust_pct_rank))
    features["deal_hold_raw"] = 0.6 * safe_divide(features["theo"], features["turnover"]).fillna(0.0) + 0.4 * safe_divide(
        features["casino_win"], features["turnover"]
    ).fillna(0.0)
    features["deal_hold_score"] = clamp_score_1_100(features.groupby("period")["deal_hold_raw"].transform(robust_pct_rank))
    features["volatility_score_behavior"] = clamp_score_1_100(
        0.30 * features.groupby("period")["max_session_loss"].transform(robust_pct_rank)
        + 0.25 * features.groupby("period")["std_bet"].transform(robust_pct_rank)
        + 0.25 * features.groupby("period")["bet_spread_ratio"].transform(robust_pct_rank)
        + 0.20 * features.groupby("period")["chase_rate"].transform(robust_pct_rank)
    )
    features = add_player_intelligence_scores(features)
    features = add_behavior_trait_scores(features)
    features["primary_behavior"] = np.select(
        [
            features["chase_rate"].ge(0.20),
            features["side_bet_rate"].ge(0.35),
            features["volatility_score_behavior"].ge(75),
            features["active_days"].ge(features.groupby("period")["active_days"].transform("median")),
        ],
        ["Chases Losses", "Side Bet Heavy", "High Volatility", "Engaged"],
        default="Balanced",
    )
    features["side_bet_intensity"] = np.select(
        [features["side_bet_rate"].ge(0.35), features["side_bet_rate"].ge(0.15)],
        ["High", "Medium"],
        default="Low",
    )
    features["risk_volatility_label"] = np.select(
        [features["volatility_score_behavior"].ge(75), features["volatility_score_behavior"].ge(50)],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )
    features["engagement_label"] = np.select(
        [features["active_days"].ge(10), features["active_days"].ge(4)],
        ["High Engagement", "Medium Engagement"],
        default="Low Engagement",
    )
    features["cohort_label"] = features["engagement_label"] + " / " + features["primary_behavior"]
    features = add_trend_features(features)
    features = features.rename(columns={"PlayerId": "player_id"})
    return features, winner_loser_ratio


def add_player_intelligence_scores(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    by_period = out.groupby("period")
    out["adt_proxy"] = safe_divide(out["theo"], out["active_days"]).fillna(0.0)
    out["range_width_score"] = by_period["bet_spread_ratio"].transform(robust_pct_rank)
    out["ceiling_pressure_score"] = by_period["ceiling_pressure_rate"].transform(robust_pct_rank)
    out["stretch_ratio"] = safe_divide(out["max_bet"], out["avg_bet"]).fillna(0.0)
    out["stretch_capacity_score"] = (
        0.55 * by_period["stretch_ratio"].transform(robust_pct_rank)
        + 0.25 * out["range_width_score"]
        + 0.20 * out["ceiling_pressure_score"]
    )
    out["rhythm_score"] = (
        0.35 * by_period["active_days"].transform(robust_pct_rank)
        + 0.30 * by_period["num_sessions"].transform(robust_pct_rank)
        + 0.25 * by_period["hours_played"].transform(robust_pct_rank)
        + 0.10 * by_period["session_continuation_rate"].transform(robust_pct_rank)
    )
    out["session_fade_score"] = (
        0.45 * by_period["post_loss_stop_rate"].transform(robust_pct_rank)
        + 0.35 * by_period["loss_exit_rate"].transform(robust_pct_rank)
        + 0.20 * (100.0 - by_period["avg_session_hours"].transform(robust_pct_rank))
    )
    out["post_loss_response_score"] = (
        0.45 * by_period["chase_rate"].transform(robust_pct_rank)
        + 0.35 * by_period["post_loss_stop_rate"].transform(robust_pct_rank)
        + 0.20 * out["volatility_score_behavior"]
    )
    out["game_affinity_score"] = (
        0.50 * out["game_concentration_pct"].clip(0, 100)
        + 0.30 * by_period["side_bet_rate"].transform(robust_pct_rank)
        + 0.20 * out["side_handle_pct"].clip(0, 100)
    )
    out["table_fit_score"] = (
        0.45 * out["game_concentration_pct"].clip(0, 100)
        + 0.25 * out["ceiling_pressure_score"]
        + 0.20 * out["range_width_score"]
        + 0.10 * by_period["hours_played"].transform(robust_pct_rank)
    )
    out["hidden_opportunity_score"] = (
        0.35 * out["rhythm_score"]
        + 0.25 * out["stretch_capacity_score"]
        + 0.20 * out["range_width_score"]
        + 0.20 * (100.0 - by_period["adt_proxy"].transform(robust_pct_rank))
    )
    out["confidence_need_score"] = (
        0.35 * out["stretch_capacity_score"]
        + 0.25 * out["range_width_score"]
        + 0.25 * out["session_fade_score"]
        + 0.15 * out["post_loss_response_score"]
    )
    score_cols = [c for c in out.columns if c.endswith("_score")]
    out[score_cols] = out[score_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 100)
    return out


def add_behavior_trait_scores(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    by_period = out.groupby("period")

    chase_rank = by_period["chase_rate"].transform(robust_pct_rank)
    stop_rank = by_period["post_loss_stop_rate"].transform(robust_pct_rank)
    side_bet_rank = by_period["side_bet_rate"].transform(robust_pct_rank)
    avg_session_rank = by_period["avg_session_hours"].transform(robust_pct_rank)
    max_session_rank = by_period["max_session_hours"].transform(robust_pct_rank)
    hours_rank = by_period["hours_played"].transform(robust_pct_rank)

    side_bet_score = 0.70 * side_bet_rank + 0.30 * out["side_handle_pct"].clip(0, 100)
    range_score = (
        0.45 * out["range_width_score"]
        + 0.35 * out["stretch_capacity_score"]
        + 0.20 * out["ceiling_pressure_score"]
    )
    stable_after_loss = 100.0 - (
        0.45 * chase_rank
        + 0.30 * stop_rank
        + 0.25 * out["volatility_score_behavior"]
    )
    stable_bet_size = 100.0 - (0.55 * out["volatility_score_behavior"] + 0.45 * out["range_width_score"])
    session_endurance = 0.45 * avg_session_rank + 0.35 * hours_rank + 0.20 * max_session_rank
    game_preference = 0.65 * out["primary_game_pct"].clip(0, 100) + 0.35 * out["game_concentration_pct"].clip(0, 100)

    out["trait_chases_after_losses_score"] = chase_rank
    out["trait_stops_quickly_after_losses_score"] = stop_rank
    out["trait_stabilizes_after_losses_score"] = stable_after_loss
    out["trait_regular_side_bet_player_score"] = side_bet_score
    out["trait_avoids_side_bets_score"] = 100.0 - side_bet_score
    out["trait_expands_bet_range_score"] = range_score
    out["trait_keeps_bet_size_stable_score"] = stable_bet_size
    out["trait_returns_frequently_score"] = out["rhythm_score"]
    out["trait_infrequent_return_pattern_score"] = 100.0 - out["rhythm_score"]
    out["trait_fades_late_in_session_score"] = out["session_fade_score"]
    out["trait_sustains_longer_sessions_score"] = session_endurance
    out["trait_strong_game_preference_score"] = game_preference
    out["trait_game_flexible_player_score"] = 100.0 - out["game_concentration_pct"].clip(0, 100)

    trait_score_cols = [f"trait_{key}_score" for key in BEHAVIOR_TRAIT_LABELS]
    out[trait_score_cols] = out[trait_score_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 100)
    best_trait_col = out[trait_score_cols].idxmax(axis=1)
    out["primary_behavior_trait_key"] = best_trait_col.str.removeprefix("trait_").str.removesuffix("_score")
    out["primary_behavior_trait_label"] = out["primary_behavior_trait_key"].map(BEHAVIOR_TRAIT_LABELS).fillna("Balanced")
    out["primary_behavior_trait_score"] = out[trait_score_cols].max(axis=1)
    return out


def add_trend_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    wide_cols = ["worth_score", "frequency_score", "volatility_score_behavior", "chase_rate", "side_bet_rate", "hours_played"]
    week = out[out["period"].isin(["Week 1", "Week 2", "Week 3"])][["PlayerId", "period"] + wide_cols].copy()
    piv = week.pivot(index="PlayerId", columns="period", values=wide_cols)
    trend = pd.DataFrame(index=piv.index)
    for col in wide_cols:
        w1 = piv.get((col, "Week 1"))
        w3 = piv.get((col, "Week 3"))
        if w1 is not None and w3 is not None:
            trend[f"{col}_trend_w3_vs_w1"] = w3 - w1
    trend = trend.reset_index()
    out = out.merge(trend, on="PlayerId", how="left")
    return out


def build_features(base_path: Path, output_dir: Path) -> dict[str, object]:
    ensure_dir(output_dir)
    session_path = base_path / SESSION_FILE
    bet_path = base_path / BET_FILE
    game_path = base_path / GAME_FILE
    print("Loading sessions...")
    sessions = prepare_sessions(session_path)
    print("Loading games...")
    games = prepare_games(game_path)
    print("Loading bets and joining context...")
    bets = prepare_bets(bet_path, games, sessions)
    print("Computing sequence features...")
    bets = add_bet_sequence_features(bets)
    print("Aggregating player-period features...")
    features, worth_ratio = build_player_period_features(sessions, bets)

    period_path = output_dir / "player_period_features.csv"
    total_path = output_dir / "player_total_features.csv"
    safe_write_csv(features, period_path)
    safe_write_csv(features[features["period"].eq("Total")].copy(), total_path)
    meta = {
        "base_path": str(base_path),
        "session_csv": str(session_path),
        "bet_csv": str(bet_path),
        "game_csv": str(game_path),
        "player_period_rows": int(len(features)),
        "player_total_rows": int(features["period"].eq("Total").sum()),
        "winner_loser_ratio": worth_ratio,
        "outputs": {"player_period_features": str(period_path), "player_total_features": str(total_path)},
    }
    write_json(meta, output_dir / "feature_metadata.json")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player features from t_bet/t_session/t_game.")
    parser.add_argument("--base-path", type=normalize_path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = build_features(args.base_path, args.output_dir)
    print(meta)


if __name__ == "__main__":
    main()
