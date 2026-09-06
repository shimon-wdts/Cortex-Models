from __future__ import annotations

from pathlib import Path

import yaml


def test_all_cohort_models_use_standard_source_tables() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    registry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model_registry"]

    for model_name in ("cohort", "cohort_tier_lift", "playerscore"):
        model = registry["models"][model_name]
        queries = {query["name"]: query["sql"] for query in model["queries"]}
        combined_sql = "\n".join(queries.values())

        assert "Crowne" not in combined_sql
        assert "FROM t_game" in queries["games"]
        assert "FROM t_game" in queries["bets"]
        assert "FROM t_bet b" in queries["bets"]
        assert 'b.theo_win AS "BetTheoWin"' in queries["bets"]
        assert "FROM t_game" in queries["sessions"]
        assert "FROM t_bet b" in queries["sessions"]
        assert "FROM t_session s" in queries["sessions"]
        assert model["kafka"]["topic"] == "model-service-insights"
        assert model["parameters"]["gaming_day_end"] == "now"
        assert model["parameters"]["observation_days"] == 21
        assert model["parameters"]["limit_games"] == 0
        assert model["parameters"]["limit_bets"] == 0
        assert model["parameters"]["json_limit"] == 0


def test_local_cohort_window_has_all_three_model_weeks_without_caps() -> None:
    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.local-env.yaml"
    registry = yaml.safe_load(settings_path.read_text(encoding="utf-8"))["model_registry"]

    for model_name in ("cohort", "cohort_tier_lift", "playerscore"):
        parameters = registry["models"][model_name]["parameters"]
        assert "gaming_day_start" not in parameters
        assert parameters["gaming_day_end"] == "now"
        assert parameters["observation_days"] == 21
        assert parameters["limit_games"] == 0
        assert parameters["limit_bets"] == 0
        assert parameters["json_limit"] == 0
