from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT
DEFAULT_BASE_PATH = PACKAGE_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "models"
CATBOOST_DEPS = PACKAGE_ROOT / "catboost_deps"

SESSION_FILE = "t_session_feb2026.csv"
BET_FILE = "t_bet_feb2026.csv"
GAME_FILE = "t_game_feb2026.csv"

PERIODS = {
    "Week 1": (1, 7),
    "Week 2": (8, 14),
    "Week 3": (15, 21),
}
PERIOD_ORDER = ["Week 1", "Week 2", "Week 3", "Total"]

MIN_BETS = 30
RANDOM_SEED = 42

BEHAVIOR_COHORT_FEATURES = [
    "active_days",
    "num_sessions",
    "hours_played",
    "avg_session_hours",
    "session_continuation_rate",
    "chase_rate",
    "loss_exit_rate",
    "post_loss_stop_rate",
    "volatility_score_behavior",
    "bet_spread_ratio",
    "side_bet_rate",
    "side_handle_pct",
    "baccarat_engagement_pct",
    "blackjack_engagement_pct",
    "game_concentration_pct",
    "ceiling_pressure_score",
    "range_width_score",
    "stretch_capacity_score",
    "rhythm_score",
    "session_fade_score",
    "post_loss_response_score",
    "game_affinity_score",
    "table_fit_score",
    "confidence_need_score",
]

CATEGORICAL_FEATURES = [
    "period",
    "primary_game",
    "primary_behavior",
    "side_bet_intensity",
    "risk_volatility_label",
    "engagement_label",
    "cohort_label",
    "cohort_model_label",
]

RECOMMENDATION_FEATURES = [
    *BEHAVIOR_COHORT_FEATURES,
    "cohort_id",
    "cohort_distance",
    "cohort_confidence_score",
    "worth_raw",
    "theo",
    "turnover",
    "avg_bet",
    "cash_buy_in",
    "credit_line",
    "hidden_opportunity_score",
]

IN_COHORT_OPTIMIZATION_CONFIG = {
    "opportunity_weight": 0.55,
    "distance_weight": 0.45,
    "min_distance_pct_in_cohort": 0.55,
    "min_opportunity_score": 60.0,
    "min_engagement_lift_prob": 0.30,
    "max_rank_offset_from_lowest": 2.0,
}

BETA_PACKAGE_CONFIG = {
    "default_top_n": 200,
    "default_per_path_cap": 75,
    "included_statuses": [
        "ready_theo_lift",
        "ready_in_cohort_optimization",
        "ready_risk_control",
        "directional_beta",
    ],
    "status_weight": {
        "ready_theo_lift": 1.00,
        "ready_in_cohort_optimization": 0.92,
        "ready_risk_control": 0.82,
        "directional_beta": 0.72,
    },
    "rank_weights": {
        "status_weight": 0.45,
        "path_fit_score": 0.30,
        "positive_outcome_support": 0.20,
        "hidden_opportunity_score": 0.05,
    },
    "positive_outcome_weights": {
        "theo_positive_rate": 0.40,
        "engagement_lift_rate": 0.30,
        "cohort_moved_up_rate": 0.15,
        "cohort_centered_rate": 0.15,
    },
}

BETA_STATUS_THRESHOLDS = {
    "min_validation_rows": 30,
    "ready_theo_lift_min_theo_ci_low": 0.0,
    "ready_theo_lift_min_engagement_ci_low": 0.50,
    "directional_theo_lift_min_avg_theo": 0.0,
    "directional_theo_lift_min_engagement_ci_low": 0.45,
    "ready_risk_control_min_risk_ci_low": 0.50,
    "ready_in_cohort_min_theo_ci_low": 0.0,
    "ready_in_cohort_min_engagement_ci_low": 0.50,
    "ready_in_cohort_min_centered_rate": 0.30,
    "directional_in_cohort_min_avg_theo": 0.0,
    "directional_in_cohort_min_engagement_ci_low": 0.35,
}

BETA_PATH_CATEGORIES = {
    "Develop from lowest cohort": "development_lift",
    "Strengthen current cohort position": "in_cohort_optimization",
    "Improve table fit and access": "fit_lift",
    "Expose to preferred game features": "engagement_lift",
    "Protect momentum after losses": "risk_control",
    "Invite to higher-limit path": "higher_limit_review",
    "Build confidence and engagement": "engagement_stability",
    "Move toward adjacent better cohort": "cohort_movement",
    "Maintain current trajectory": "monitor",
}

BETA_PATH_UI_BADGES = {
    "development_lift": ["Development", "Theo supported"],
    "in_cohort_optimization": ["In-cohort", "Centering"],
    "fit_lift": ["Fit", "Theo supported"],
    "engagement_lift": ["Engagement", "Directional"],
    "risk_control": ["Risk control", "Operator review"],
    "higher_limit_review": ["Higher limit", "Hold"],
    "engagement_stability": ["Stability", "Review"],
    "cohort_movement": ["Cohort move", "Review"],
    "monitor": ["Monitor"],
}

RESPONSIBLE_GAMING_REVIEW_CONFIG = {
    "mode": "caution_only_not_enforced",
    "review_required": True,
    "blocks_recommendation": False,
    "ranking_enforced": False,
    "note": "Product responsible-gaming policy is pending; apply partner policy before acting on any recommendation.",
}

FEEDBACK_TEMPLATE_CONFIG = {
    "prefill_columns": [
        "player_id",
        "recommended_path",
        "path_category",
        "beta_status",
        "beta_action",
        "beta_rank_score",
        "path_fit_score",
        "primary_outcome_metric",
        "primary_outcome_mean",
        "primary_outcome_ci_low",
        "primary_outcome_ci_high",
        "primary_outcome_success_rate",
        "responsible_gaming_review_required",
    ],
    "feedback_columns": [
        "shown_to_operator",
        "operator_decision",
        "rejection_reason",
        "action_taken",
        "action_type",
        "action_date",
        "operator_notes",
        "followup_date",
        "observed_outcome_window",
        "observed_theo_delta",
        "observed_turnover_delta",
        "observed_engagement_delta",
        "observed_cohort_change",
        "observed_risk_change",
        "outcome_notes",
    ],
    "allowed_values": {
        "shown_to_operator": ["yes", "no"],
        "operator_decision": ["accepted", "rejected", "deferred", "needs_more_context"],
        "action_taken": ["yes", "no", "pending"],
        "observed_outcome_window": ["7_day", "14_day", "30_day", "custom"],
        "observed_cohort_change": ["moved_up", "centered_same_cohort", "no_change", "moved_down", "unknown"],
        "observed_risk_change": ["improved", "unchanged", "worsened", "unknown"],
    },
}
