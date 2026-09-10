from __future__ import annotations

from pathlib import Path

import yaml


def test_shoe_advantage_publishes_to_table_guard_topic() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    registry = yaml.safe_load(config_path.read_text(encoding="utf-8"))["model_registry"]

    shoe_advantage = registry["models"]["ShoeAdvantage"]

    assert shoe_advantage["kafka"] == {
        "topic": "table-guard-insights",
        "key_field": "shoe_id",
    }
