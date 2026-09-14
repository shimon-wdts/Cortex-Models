from __future__ import annotations

from pathlib import Path

import yaml


def test_all_cohort_models_use_standard_source_tables() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    registry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model_registry"]

    for model_name in ("cohort", "cohort_tier_lift", "playerscore"):
        model = registry["models"][model_name]
        queries = {query["name"]: query["sql"] for query in model["queries"]}
        query_configs = {query["name"]: query for query in model["queries"]}
        combined_sql = "\n".join(queries.values())

        assert "Crowne" not in combined_sql
        assert "HAVING COUNT(*) >= :minimum_bets" in queries["eligible_players"]
        assert "FROM t_game" in queries["games"]
        assert "ANY(:player_ids)" in queries["games"]
        assert "LIMIT" not in combined_sql
        assert "ORDER BY" not in combined_sql
        assert "JOIN t_game" in queries["bets"]
        assert "FROM t_bet b" in queries["bets"]
        assert 'b.theo_win AS "BetTheoWin"' in queries["bets"]
        assert "JOIN t_game" in queries["sessions"]
        assert "FROM t_bet b" in queries["sessions"]
        assert "FROM t_session s" in queries["sessions"]
        for query_name in ("games", "bets", "sessions"):
            assert query_configs[query_name]["batch_source_query"] == "eligible_players"
            assert query_configs[query_name]["batch_source_column"] == "PlayerId"
            assert query_configs[query_name]["batch_param"] == "player_ids"
        assert model["kafka"]["topic"] == "model-service-insights"
        assert model["parameters"]["gaming_day_end"] == "now"
        assert model["parameters"]["observation_days"] == 21
        assert model["parameters"]["minimum_bets"] == 30
        assert model["parameters"]["extract_batch_size"] == 500
        assert model["parameters"]["extract_workers"] == 4
        assert model["parameters"]["json_limit"] == 0


def test_local_cohort_window_has_all_three_model_weeks_without_output_cap() -> None:
    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.local-env.yaml"
    registry = yaml.safe_load(settings_path.read_text(encoding="utf-8"))["model_registry"]

    for model_name in ("cohort", "cohort_tier_lift", "playerscore"):
        parameters = registry["models"][model_name]["parameters"]
        assert "gaming_day_start" not in parameters
        assert parameters["gaming_day_end"] == "now"
        assert parameters["observation_days"] == 21
        assert parameters["minimum_bets"] == 30
        assert parameters["extract_batch_size"] == 500
        assert parameters["extract_workers"] == 4
        assert parameters["json_limit"] == 0
