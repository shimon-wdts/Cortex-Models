from __future__ import annotations

from pathlib import Path

import yaml


def test_shoe_advantage_publishes_to_model_service_topic() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    registry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model_registry"]

    shoe_advantage = registry["models"]["ShoeAdvantage"]

    assert shoe_advantage["kafka"] == {
        "topic": "model-service-insights",
        "key_field": "shoe_id",
    }


def test_shoe_advantage_query_scopes_replayed_history_to_a_shoe_instance() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    registry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model_registry"]
    query = registry["models"]["ShoeAdvantage"]["queries"][0]["sql"]
    compact_sql = " ".join(query.split())

    assert "SELECT DISTINCT gaming_day, table_id, shoe_id" in compact_sql
    assert "active.gaming_day = g.gaming_day" in compact_sql
    assert "active.table_id = g.table_id" in compact_sql
    assert "active.shoe_id = g.shoe_id" in compact_sql
    assert "ORDER BY g.gaming_day, g.table_id, g.shoe_id, g.shoe_game_count" in compact_sql
