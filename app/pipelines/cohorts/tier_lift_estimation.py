from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class HistoricalGrowthPrior:
    behavior_change: str
    median_awt_growth_pct: float
    median_adt_growth_pct: float
    median_awt_growth_ci95: tuple[float, float]
    players: int


@dataclass(frozen=True)
class PersonalizedTierLift:
    current_theo: float
    expected_theo: float
    expected_theo_lift: float
    predicted_growth_pct: float | None
    range95: tuple[float, float]
    visits_per_week: float
    frequency_factor: float
    recency_days: int | None
    recency_factor: float
    path_fit_factor: float
    historical_behavior_change: str
    historical_median_awt_growth_pct: float
    historical_median_adt_growth_pct: float
    historical_players: int
    method: str = "historical_behavior_frequency_recency_v1"

    def diagnostics(self) -> dict[str, Any]:
        values = asdict(self)
        return {
            "method": values["method"],
            "historical_behavior_change": values["historical_behavior_change"],
            "historical_median_awt_growth_pct": values["historical_median_awt_growth_pct"],
            "historical_median_adt_growth_pct": values["historical_median_adt_growth_pct"],
            "historical_players": values["historical_players"],
            "visits_per_week": values["visits_per_week"],
            "frequency_factor": values["frequency_factor"],
            "recency_days": values["recency_days"],
            "recency_factor": values["recency_factor"],
            "path_fit_factor": values["path_fit_factor"],
            "outcome_definition": "weekly Theo (AWT)",
            "historical_window": "Crowne 2025-10 through 2025-11",
            "range95_method": "scaled bootstrap CI95 for historical median AWT growth",
        }


# Crowne October-November 2025 behavior-adoption backtest. The confidence
# intervals are bootstrap 95% confidence intervals for the median AWT growth.
HISTORICAL_GROWTH_PRIORS = {
    "Increase return rhythm": HistoricalGrowthPrior(
        "Increase return rhythm", 96.0, 4.1, (73.9, 108.5), 477
    ),
    "Build confidence and engagement": HistoricalGrowthPrior(
        "Build confidence and engagement", 105.4, 5.8, (81.7, 120.4), 185
    ),
    "Invite to higher-limit path": HistoricalGrowthPrior(
        "Invite to higher-limit path", 94.2, 17.1, (53.5, 144.2), 133
    ),
    "Expose to preferred game features": HistoricalGrowthPrior(
        "Expose to preferred game features", 89.0, 19.6, (46.1, 166.6), 110
    ),
    "Improve table fit and access": HistoricalGrowthPrior(
        "Improve table fit and access", 13.4, -7.0, (-16.9, 43.4), 110
    ),
    "Protect momentum after losses": HistoricalGrowthPrior(
        "Protect momentum after losses", -10.3, -30.5, (-21.3, 5.1), 325
    ),
    "Maintain current trajectory": HistoricalGrowthPrior(
        "Maintain current trajectory", 0.0, 0.0, (0.0, 0.0), 0
    ),
}

PATH_PRIOR_ALIASES = {
    "Strengthen current cohort position": "Improve table fit and access",
    "Develop hidden opportunity": "Increase return rhythm",
    "Develop from lowest cohort": "Build confidence and engagement",
    "Move toward adjacent better cohort": "Improve table fit and access",
    "Player development path": "Build confidence and engagement",
}


def estimate_personalized_tier_lift(row: pd.Series) -> PersonalizedTierLift:
    path = _text(row.get("recommended_path"), "Player development path")
    prior_name = PATH_PRIOR_ALIASES.get(path, path)
    prior = HISTORICAL_GROWTH_PRIORS.get(prior_name, HISTORICAL_GROWTH_PRIORS["Improve table fit and access"])

    current_awt = _current_awt(row)
    visits_per_week = _visits_per_week(row)
    frequency_factor = round(_clip(1.0 - visits_per_week / 4.0, 0.25, 1.0), 6)
    recency_days = _recency_days(row)
    recency_factor = round(_clip(exp(-recency_days / 30.0), 0.50, 1.0), 6) if recency_days is not None else 1.0
    path_fit_factor = round(0.75 + 0.25 * _path_fit_probability(row), 6)

    growth_pct = _personalized_growth(prior.median_awt_growth_pct, prior.median_adt_growth_pct, frequency_factor)
    low_growth_pct = _personalized_growth(
        prior.median_awt_growth_ci95[0], prior.median_adt_growth_pct, frequency_factor
    )
    high_growth_pct = _personalized_growth(
        prior.median_awt_growth_ci95[1], prior.median_adt_growth_pct, frequency_factor
    )
    growth_pct *= recency_factor * path_fit_factor
    low_growth_pct *= recency_factor * path_fit_factor
    high_growth_pct *= recency_factor * path_fit_factor

    if current_awt <= 0:
        predicted_growth_pct = None
        expected_lift = 0.0
        expected_theo = 0.0
        lift_range = (0.0, 0.0)
    else:
        predicted_growth_pct = round(growth_pct, 4)
        expected_lift = round(current_awt * growth_pct / 100.0, 2)
        expected_theo = round(max(0.0, current_awt + expected_lift), 2)
        lift_range = tuple(
            sorted(
                (
                    round(current_awt * low_growth_pct / 100.0, 2),
                    round(current_awt * high_growth_pct / 100.0, 2),
                )
            )
        )

    return PersonalizedTierLift(
        current_theo=round(current_awt, 2),
        expected_theo=expected_theo,
        expected_theo_lift=expected_lift,
        predicted_growth_pct=predicted_growth_pct,
        range95=lift_range,
        visits_per_week=round(visits_per_week, 2),
        frequency_factor=frequency_factor,
        recency_days=recency_days,
        recency_factor=recency_factor,
        path_fit_factor=path_fit_factor,
        historical_behavior_change=prior.behavior_change,
        historical_median_awt_growth_pct=prior.median_awt_growth_pct,
        historical_median_adt_growth_pct=prior.median_adt_growth_pct,
        historical_players=prior.players,
    )


def _current_awt(row: pd.Series) -> float:
    explicit_awt = _number(row.get("current_awt"), None)
    if explicit_awt is not None:
        return max(0.0, explicit_awt)
    total_theo = _number(row.get("total_session_theo"), None)
    if total_theo is None:
        total_theo = _number(row.get("theo"), 0.0) or 0.0
    observed_weeks = _number(row.get("observation_weeks"), 3.0) or 3.0
    return max(0.0, total_theo / max(observed_weeks, 1.0))


def _visits_per_week(row: pd.Series) -> float:
    active_days = _number(row.get("active_days"), None)
    if active_days is None or active_days <= 0:
        active_days = _number(row.get("num_sessions"), None)
    if active_days is None or active_days <= 0:
        return 1.0
    observed_weeks = _number(row.get("observation_weeks"), 3.0) or 3.0
    return max(0.0, active_days / max(observed_weeks, 1.0))


def _recency_days(row: pd.Series) -> int | None:
    last_day = pd.to_datetime(row.get("last_gaming_day"), errors="coerce")
    if pd.isna(last_day):
        return None
    reference_day = pd.to_datetime(row.get("observation_end_day"), errors="coerce")
    if pd.isna(reference_day):
        reference_day = last_day.replace(day=21).normalize()
    return max(0, int((reference_day - last_day.normalize()).days))


def _path_fit_probability(row: pd.Series) -> float:
    raw_path_fit = _number(row.get("path_fit_score"), None)
    if raw_path_fit is None:
        raw_path_fit = _number(row.get("recommendation_confidence"), 0.8) or 0.8
    if raw_path_fit > 1.0:
        raw_path_fit /= 100.0
    return _clip(raw_path_fit, 0.0, 1.0)


def _personalized_growth(awt_growth_pct: float, adt_growth_pct: float, frequency_factor: float) -> float:
    frequency_component = awt_growth_pct - adt_growth_pct
    return adt_growth_pct + frequency_factor * frequency_component


def _number(value: Any, default: float | None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any, default: str) -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text or default


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
