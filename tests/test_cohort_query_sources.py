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
        assert model["parameters"]["gaming_day_start"] == "2026-07-01"
        assert model["parameters"]["gaming_day_end"] == "2026-07-21"


def test_uat_cohort_window_has_all_three_model_weeks() -> None:
    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.local.yaml"
    registry = yaml.safe_load(settings_path.read_text(encoding="utf-8"))["model_registry"]

    for model_name in ("cohort", "cohort_tier_lift", "playerscore"):
        parameters = registry["models"][model_name]["parameters"]
        assert parameters["gaming_day_start"] == "2026-07-01"
        assert parameters["gaming_day_end"] == "2026-07-21"
