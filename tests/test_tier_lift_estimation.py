from __future__ import annotations

import pandas as pd

from app.pipelines.cohorts.tier_lift_estimation import estimate_personalized_tier_lift


def test_same_path_produces_personalized_growth_by_frequency_and_recency() -> None:
    common = {
        "recommended_path": "Increase return rhythm",
        "total_session_theo": 900,
        "path_fit_score": 1.0,
    }
    low_frequency_recent = estimate_personalized_tier_lift(
        pd.Series({**common, "active_days": 3, "last_gaming_day": "2026-07-21"})
    )
    high_frequency_stale = estimate_personalized_tier_lift(
        pd.Series({**common, "active_days": 9, "last_gaming_day": "2026-07-01"})
    )

    assert low_frequency_recent.current_theo == 300
    assert low_frequency_recent.predicted_growth_pct == 73.025
    assert low_frequency_recent.expected_theo_lift == 219.07
    assert low_frequency_recent.range95 == (169.35, 247.2)
    assert high_frequency_stale.current_theo == 300
    assert high_frequency_stale.predicted_growth_pct == 13.9008
    assert high_frequency_stale.expected_theo_lift == 41.7
    assert low_frequency_recent.predicted_growth_pct > high_frequency_stale.predicted_growth_pct


def test_explicit_current_awt_is_not_divided_again() -> None:
    estimate = estimate_personalized_tier_lift(
        pd.Series(
            {
                "recommended_path": "Invite to higher-limit path",
                "current_awt": 1000,
                "active_days": 3,
                "last_gaming_day": "2026-07-21",
                "path_fit_score": 1.0,
            }
        )
    )
    assert estimate.current_theo == 1000
    assert estimate.expected_theo_lift == 749.25


def test_protect_momentum_does_not_promise_positive_growth() -> None:
    estimate = estimate_personalized_tier_lift(
        pd.Series(
            {
                "recommended_path": "Protect momentum after losses",
                "total_session_theo": 3000,
                "active_days": 3,
                "last_gaming_day": "2026-07-21",
                "path_fit_score": 1.0,
            }
        )
    )
    assert estimate.predicted_growth_pct < 0
    assert estimate.expected_theo_lift < 0
