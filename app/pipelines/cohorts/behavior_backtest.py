from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BehaviorThresholds:
    """Material-change thresholds used to identify historical recommendation analogs."""

    rhythm_days_per_week_delta: float = 0.5
    rhythm_sessions_per_week_delta: float = 1.0
    preferred_share_delta: float = 0.10
    upper_wager_ratio: float = 1.10
    upper_share_delta: float = 0.05
    confidence_hours_ratio: float = 1.10
    allowed_chase_increase: float = 0.02
    allowed_volatility_ratio: float = 1.10
    momentum_issue_chase_rate: float = 0.10
    momentum_issue_wager_cv: float = 1.00
    momentum_chase_reduction: float = 0.05
    momentum_volatility_ratio: float = 0.90
    side_bet_baseline_rate: float = 0.05
    side_bet_rate_delta: float = 0.10
    min_future_bets_for_behavior: int = 10


BEHAVIOR_COLUMNS = {
    "Increase return rhythm": "followed_increase_return_rhythm",
    "Improve table fit and access": "followed_improve_table_fit",
    "Invite to higher-limit path": "followed_higher_limit_path",
    "Build confidence and engagement": "followed_build_confidence",
    "Protect momentum after losses": "followed_protect_momentum",
    "Expose to preferred game features": "followed_preferred_features",
}

MATCH_FEATURES = [
    "baseline_awt",
    "baseline_adt",
    "baseline_active_days_per_week",
    "baseline_sessions_per_week",
    "baseline_hours_per_week",
    "baseline_avg_bet",
    "baseline_p90_wager",
    "baseline_preferred_game_share",
    "baseline_preferred_table_share",
    "baseline_side_bet_rate",
    "baseline_chase_rate",
    "baseline_wager_cv",
]


def run_behavior_backtest(
    merged_bets: pd.DataFrame,
    months: Iterable[str] | None = None,
    *,
    min_baseline_bets: int = 30,
    thresholds: BehaviorThresholds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Measure AWT/ADT growth for players who exhibited recommended behavior changes.

    The input is the Crowne merged bet/game/session extract. Each complete month uses
    days 1-21 as the three-week baseline and days 22-28 as the one-week evaluation
    period. AWT is the primary outcome; ADT and active gaming days are secondary.
    """

    limits = thresholds or BehaviorThresholds()
    bets = prepare_merged_bets(merged_bets)
    selected_months = list(months) if months is not None else complete_months(bets)
    if not selected_months:
        raise ValueError("No complete month contains both baseline days 1-21 and evaluation days 22-28.")

    outcomes = []
    for month in selected_months:
        month_rows = bets[bets["gaming_day"].dt.strftime("%Y-%m").eq(month)].copy()
        baseline = month_rows[month_rows["gaming_day"].dt.day.between(1, 21)].copy()
        future = month_rows[month_rows["gaming_day"].dt.day.between(22, 28)].copy()
        if baseline.empty or future.empty:
            continue
        month_outcomes = build_month_outcomes(baseline, future, limits)
        month_outcomes["month"] = month
        outcomes.append(month_outcomes)

    if not outcomes:
        raise ValueError("The requested months did not contain usable baseline and evaluation rows.")

    player_outcomes = pd.concat(outcomes, ignore_index=True)
    player_outcomes = player_outcomes[
        player_outcomes["baseline_bet_count"].ge(min_baseline_bets)
        & player_outcomes["baseline_theo"].gt(0)
    ].copy()
    summary = summarize_behavior_growth(player_outcomes)
    metadata = {
        "months": selected_months,
        "baseline_window": "calendar days 1-21 (3 weeks)",
        "evaluation_window": "calendar days 22-28 (1 week)",
        "primary_outcome": "AWT = TheoWin / observed weeks",
        "secondary_outcomes": [
            "ADT = TheoWin / active gaming days",
            "active gaming days per week",
        ],
        "minimum_baseline_bets": int(min_baseline_bets),
        "thresholds": asdict(limits),
        "behavior_columns": BEHAVIOR_COLUMNS,
        "eligible_player_months": int(len(player_outcomes)),
        "unique_players": int(player_outcomes["player_id"].nunique()),
    }
    return summary, player_outcomes, metadata


def prepare_merged_bets(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["PlayerId", "SessionId", "GamingDay", "Wager", "TheoWin"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Crowne merged extract is missing required columns: {missing}")

    data = pd.DataFrame(index=frame.index)
    data["player_id"] = clean_id(frame["PlayerId"])
    data["session_id"] = clean_id(frame["SessionId"])
    data["bet_id"] = pd.to_numeric(_series(frame, "BetId"), errors="coerce")
    data["gaming_day"] = pd.to_datetime(frame["GamingDay"], errors="coerce").dt.normalize()
    data["wager"] = pd.to_numeric(frame["Wager"], errors="coerce").fillna(0.0)
    data["theo"] = pd.to_numeric(frame["TheoWin"], errors="coerce").fillna(0.0)
    data["casino_win"] = pd.to_numeric(_series(frame, "CasinoWin"), errors="coerce").fillna(0.0)
    data["table_id"] = clean_id(_series(frame, "TableId"))
    game_type_source = "GameType_game" if "GameType_game" in frame.columns else "GameType"
    data["game_type"] = _series(frame, game_type_source).fillna("UNKNOWN").astype(str).str.upper().str.strip()
    data["bet_type"] = _series(frame, "BetType").fillna("").astype(str).str.upper()
    data["type_of_bet"] = _series(frame, "TypeOfBet").fillna("").astype(str).str.upper()
    data["event_time"] = pd.to_datetime(_series(frame, "PayoutCompleteDtm"), utc=True, errors="coerce")
    data["session_start"] = pd.to_datetime(_series(frame, "SessionStartDtm"), utc=True, errors="coerce")
    data["session_end"] = pd.to_datetime(_series(frame, "SessionEndDtm"), utc=True, errors="coerce")
    data["is_side_bet"] = (
        data["type_of_bet"].ne("MAIN_BET")
        | data["bet_type"].str.contains("PAIR|TIE|JACKPOT|SIDE|LUCKY", regex=True, na=False)
    ).astype(int)
    return data[data["player_id"].ne("") & data["gaming_day"].notna()].copy()


def complete_months(bets: pd.DataFrame) -> list[str]:
    available = bets.assign(month=bets["gaming_day"].dt.strftime("%Y-%m"))
    result = []
    for month, rows in available.groupby("month"):
        days = set(rows["gaming_day"].dt.day.unique().tolist())
        if set(range(1, 29)).issubset(days):
            result.append(str(month))
    return sorted(result)


def build_month_outcomes(
    baseline: pd.DataFrame,
    future: pd.DataFrame,
    thresholds: BehaviorThresholds,
) -> pd.DataFrame:
    baseline_metrics = aggregate_period(baseline, weeks=3.0).add_prefix("baseline_")
    future_metrics = aggregate_period(future, weeks=1.0).add_prefix("future_")
    outcomes = baseline_metrics.join(future_metrics, how="left")

    future_zero = [
        "theo",
        "awt",
        "active_days",
        "active_days_per_week",
        "session_count",
        "sessions_per_week",
        "bet_count",
        "hours",
        "hours_per_week",
        "avg_session_hours",
        "avg_bet",
        "p90_wager",
        "wager_cv",
        "side_bet_rate",
        "chase_rate",
    ]
    for column in future_zero:
        name = f"future_{column}"
        if name in outcomes:
            outcomes[name] = outcomes[name].fillna(0.0)

    preference = preference_change_features(baseline, future)
    upper_range = upper_range_features(baseline, future)
    outcomes = outcomes.join(preference, how="left").join(upper_range, how="left")
    share_columns = [
        "baseline_preferred_game_share",
        "future_preferred_game_share",
        "baseline_preferred_table_share",
        "future_preferred_table_share",
        "baseline_upper_wager_share",
        "future_upper_wager_share",
    ]
    for column in share_columns:
        outcomes[column] = outcomes[column].fillna(0.0)

    outcomes["awt_growth_pct"] = percent_growth(outcomes["future_awt"], outcomes["baseline_awt"])
    outcomes["adt_growth_pct"] = percent_growth(outcomes["future_adt"], outcomes["baseline_adt"])
    outcomes["active_days_growth_pct"] = percent_growth(
        outcomes["future_active_days_per_week"], outcomes["baseline_active_days_per_week"]
    )

    active_future = outcomes["future_active_days"].gt(0)
    enough_future_bets = outcomes["future_bet_count"].ge(thresholds.min_future_bets_for_behavior)
    outcomes[BEHAVIOR_COLUMNS["Increase return rhythm"]] = active_future & (
        outcomes["future_active_days_per_week"].sub(outcomes["baseline_active_days_per_week"]).ge(
            thresholds.rhythm_days_per_week_delta
        )
        | outcomes["future_sessions_per_week"].sub(outcomes["baseline_sessions_per_week"]).ge(
            thresholds.rhythm_sessions_per_week_delta
        )
    )
    outcomes[BEHAVIOR_COLUMNS["Improve table fit and access"]] = enough_future_bets & (
        outcomes["future_preferred_game_share"].sub(outcomes["baseline_preferred_game_share"]).ge(
            thresholds.preferred_share_delta
        )
        | outcomes["future_preferred_table_share"].sub(outcomes["baseline_preferred_table_share"]).ge(
            thresholds.preferred_share_delta
        )
    )
    outcomes[BEHAVIOR_COLUMNS["Invite to higher-limit path"]] = (
        enough_future_bets
        & outcomes["baseline_p90_wager"].gt(0)
        & outcomes["future_p90_wager"].ge(outcomes["baseline_p90_wager"] * thresholds.upper_wager_ratio)
        & outcomes["future_upper_wager_share"].sub(outcomes["baseline_upper_wager_share"]).ge(
            thresholds.upper_share_delta
        )
    )
    outcomes[BEHAVIOR_COLUMNS["Build confidence and engagement"]] = (
        enough_future_bets
        & outcomes["future_hours_per_week"].ge(
            outcomes["baseline_hours_per_week"] * thresholds.confidence_hours_ratio
        )
        & outcomes["future_active_days_per_week"].ge(outcomes["baseline_active_days_per_week"])
        & outcomes["future_chase_rate"].le(
            outcomes["baseline_chase_rate"] + thresholds.allowed_chase_increase
        )
        & outcomes["future_wager_cv"].le(
            outcomes["baseline_wager_cv"] * thresholds.allowed_volatility_ratio
        )
    )
    momentum_issue = (
        outcomes["baseline_chase_rate"].ge(thresholds.momentum_issue_chase_rate)
        | outcomes["baseline_wager_cv"].ge(thresholds.momentum_issue_wager_cv)
    )
    outcomes[BEHAVIOR_COLUMNS["Protect momentum after losses"]] = (
        enough_future_bets
        & momentum_issue
        & (
            outcomes["baseline_chase_rate"].sub(outcomes["future_chase_rate"]).ge(
                thresholds.momentum_chase_reduction
            )
            | outcomes["future_wager_cv"].le(
                outcomes["baseline_wager_cv"] * thresholds.momentum_volatility_ratio
            )
        )
    )
    outcomes[BEHAVIOR_COLUMNS["Expose to preferred game features"]] = (
        enough_future_bets
        & outcomes["baseline_side_bet_rate"].ge(thresholds.side_bet_baseline_rate)
        & outcomes["future_side_bet_rate"].sub(outcomes["baseline_side_bet_rate"]).ge(
            thresholds.side_bet_rate_delta
        )
    )
    return outcomes.reset_index().rename(columns={"index": "player_id"})


def aggregate_period(data: pd.DataFrame, *, weeks: float) -> pd.DataFrame:
    sequenced = data.sort_values(["player_id", "session_id", "event_time", "bet_id"], kind="mergesort").copy()
    sequenced["previous_wager"] = sequenced.groupby(["player_id", "session_id"])["wager"].shift(1)
    sequenced["previous_casino_win"] = sequenced.groupby(["player_id", "session_id"])["casino_win"].shift(1)
    sequenced["chase_flag"] = (
        sequenced["previous_casino_win"].gt(0) & sequenced["wager"].gt(sequenced["previous_wager"])
    ).astype(int)

    metrics = sequenced.groupby("player_id").agg(
        theo=("theo", "sum"),
        active_days=("gaming_day", "nunique"),
        session_count=("session_id", "nunique"),
        bet_count=("bet_id", "count"),
        total_wager=("wager", "sum"),
        avg_bet=("wager", "mean"),
        wager_std=("wager", "std"),
        p90_wager=("wager", lambda values: values.quantile(0.90)),
        side_bet_rate=("is_side_bet", "mean"),
        chase_rate=("chase_flag", "mean"),
    )
    sessions = sequenced.drop_duplicates(["player_id", "session_id"]).copy()
    sessions["session_hours"] = (
        (sessions["session_end"] - sessions["session_start"]).dt.total_seconds().clip(lower=0) / 3600.0
    ).fillna(0.0)
    session_metrics = sessions.groupby("player_id").agg(
        hours=("session_hours", "sum"),
        avg_session_hours=("session_hours", "mean"),
    )
    metrics = metrics.join(session_metrics, how="left")
    metrics["awt"] = metrics["theo"] / weeks
    metrics["adt"] = metrics["theo"] / metrics["active_days"].replace(0, np.nan)
    metrics["active_days_per_week"] = metrics["active_days"] / weeks
    metrics["sessions_per_week"] = metrics["session_count"] / weeks
    metrics["hours_per_week"] = metrics["hours"] / weeks
    metrics["wager_cv"] = metrics["wager_std"].fillna(0.0) / metrics["avg_bet"].replace(0, np.nan)
    metrics["wager_cv"] = metrics["wager_cv"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return metrics


def preference_change_features(baseline: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    players = pd.Index(baseline["player_id"].unique(), name="player_id")
    result = pd.DataFrame(index=players)
    for dimension, label in [("game_type", "game"), ("table_id", "table")]:
        baseline_amounts = baseline.groupby(["player_id", dimension])["wager"].sum().reset_index()
        preferred = (
            baseline_amounts.sort_values(["player_id", "wager"], ascending=[True, False])
            .drop_duplicates("player_id")
            .rename(columns={dimension: f"preferred_{label}", "wager": "preferred_wager"})
        )
        totals = baseline.groupby("player_id")["wager"].sum().rename("total_wager")
        preferred = preferred.join(totals, on="player_id")
        preferred[f"baseline_preferred_{label}_share"] = (
            preferred["preferred_wager"] / preferred["total_wager"].replace(0, np.nan)
        )
        future_amounts = future.groupby(["player_id", dimension])["wager"].sum().reset_index()
        future_totals = future.groupby("player_id")["wager"].sum().rename("future_total_wager")
        preferred = preferred.merge(
            future_amounts,
            left_on=["player_id", f"preferred_{label}"],
            right_on=["player_id", dimension],
            how="left",
        ).join(future_totals, on="player_id")
        preferred[f"future_preferred_{label}_share"] = (
            preferred["wager"].fillna(0.0) / preferred["future_total_wager"].replace(0, np.nan)
        ).fillna(0.0)
        preferred = preferred.set_index("player_id")
        result = result.join(
            preferred[
                [
                    f"baseline_preferred_{label}_share",
                    f"future_preferred_{label}_share",
                ]
            ],
            how="left",
        )
    return result


def upper_range_features(baseline: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    threshold = baseline.groupby("player_id")["wager"].quantile(0.90).rename("upper_wager_threshold")
    baseline_joined = baseline.join(threshold, on="player_id")
    future_joined = future.join(threshold, on="player_id")
    baseline_share = (
        baseline_joined["wager"].ge(baseline_joined["upper_wager_threshold"])
        .groupby(baseline_joined["player_id"])
        .mean()
        .rename("baseline_upper_wager_share")
    )
    future_share = (
        future_joined["wager"].ge(future_joined["upper_wager_threshold"])
        .groupby(future_joined["player_id"])
        .mean()
        .rename("future_upper_wager_share")
    )
    return pd.concat([baseline_share, future_share], axis=1)


def summarize_behavior_growth(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for behavior, flag in BEHAVIOR_COLUMNS.items():
        followers = outcomes[outcomes[flag].fillna(False)].copy()
        controls = matched_controls(outcomes, flag)
        returning_controls = matched_controls(outcomes, flag, require_future_activity=True)
        follower_metrics = growth_metrics(followers)
        control_metrics = growth_metrics(controls)
        returning_control_metrics = growth_metrics(returning_controls)
        rows.append(
            {
                "behavior_change": behavior,
                "followers": int(len(followers)),
                "matched_control_rows": int(len(controls)),
                "unique_matched_controls": int(controls["player_id"].nunique()) if not controls.empty else 0,
                "unique_matched_returning_controls": (
                    int(returning_controls["player_id"].nunique()) if not returning_controls.empty else 0
                ),
                **{f"followers_{key}": value for key, value in follower_metrics.items()},
                **{f"matched_controls_{key}": value for key, value in control_metrics.items()},
                **{
                    f"matched_returning_controls_{key}": value
                    for key, value in returning_control_metrics.items()
                },
                "median_awt_growth_difference_pp": follower_metrics["median_awt_growth_pct"]
                - control_metrics["median_awt_growth_pct"],
                "portfolio_awt_growth_difference_pp": follower_metrics["portfolio_awt_growth_pct"]
                - control_metrics["portfolio_awt_growth_pct"],
                "median_awt_growth_vs_returning_controls_pp": follower_metrics["median_awt_growth_pct"]
                - returning_control_metrics["median_awt_growth_pct"],
                "portfolio_awt_growth_vs_returning_controls_pp": follower_metrics["portfolio_awt_growth_pct"]
                - returning_control_metrics["portfolio_awt_growth_pct"],
            }
        )
    return pd.DataFrame(rows).sort_values("followers", ascending=False, kind="mergesort").reset_index(drop=True)


def matched_controls(
    outcomes: pd.DataFrame,
    behavior_flag: str,
    *,
    require_future_activity: bool = False,
) -> pd.DataFrame:
    matched = []
    for month, month_rows in outcomes.groupby("month"):
        followers = month_rows[month_rows[behavior_flag].fillna(False)].copy()
        controls = month_rows[~month_rows[behavior_flag].fillna(False)].copy()
        if require_future_activity:
            controls = controls[controls["future_active_days"].gt(0)].copy()
        if followers.empty or controls.empty:
            continue
        combined = pd.concat([followers[MATCH_FEATURES], controls[MATCH_FEATURES]], ignore_index=True)
        combined = combined.apply(pd.to_numeric, errors="coerce")
        medians = combined.median().fillna(0.0)
        combined = combined.fillna(medians)
        scaled = StandardScaler().fit_transform(np.log1p(combined.clip(lower=0.0)))
        follower_matrix = scaled[: len(followers)]
        control_matrix = scaled[len(followers) :]
        neighbor = NearestNeighbors(n_neighbors=1).fit(control_matrix)
        _, indexes = neighbor.kneighbors(follower_matrix)
        selected = controls.iloc[indexes[:, 0]].copy()
        selected["matched_for_month"] = month
        matched.append(selected)
    if not matched:
        return outcomes.iloc[0:0].copy()
    return pd.concat(matched, ignore_index=True)


def growth_metrics(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        return {
            "median_awt_growth_pct": np.nan,
            "portfolio_awt_growth_pct": np.nan,
            "median_adt_growth_pct": np.nan,
            "median_active_days_growth_pct": np.nan,
            "return_rate": np.nan,
        }
    baseline_awt = rows["baseline_awt"].sum()
    portfolio_growth = (
        (rows["future_awt"].sum() / baseline_awt - 1.0) * 100.0 if baseline_awt > 0 else np.nan
    )
    return {
        "median_awt_growth_pct": float(rows["awt_growth_pct"].median()),
        "portfolio_awt_growth_pct": float(portfolio_growth),
        "median_adt_growth_pct": float(rows["adt_growth_pct"].median()),
        "median_active_days_growth_pct": float(rows["active_days_growth_pct"].median()),
        "return_rate": float(rows["future_active_days"].gt(0).mean()),
    }


def percent_growth(future: pd.Series, baseline: pd.Series) -> pd.Series:
    return (future / baseline.replace(0, np.nan) - 1.0) * 100.0


def clean_id(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip()


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=frame.index)
