from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import app.clients.kafka as kafka_module
from app.clients.kafka import KafkaPredictionPublisher


class FakeProducer:
    def __init__(self, _config: dict[str, Any]) -> None:
        self.messages: list[dict[str, Any]] = []

    def produce(self, **message: Any) -> None:
        self.messages.append(message)

    def poll(self, _timeout: float) -> None:
        return None

    def flush(self) -> None:
        return None


@pytest.fixture
def fake_producer(monkeypatch: pytest.MonkeyPatch) -> FakeProducer:
    producer = FakeProducer({})
    module = SimpleNamespace(Producer=lambda _config: producer)
    monkeypatch.setitem(sys.modules, "confluent_kafka", module)
    return producer


def test_publisher_sanitizes_nested_nan_and_emits_strict_json(fake_producer: FakeProducer) -> None:
    publisher = KafkaPredictionPublisher(bootstrap_servers="kafka:9092")
    event = {
        "event_id": "event-1",
        "score": float("nan"),
        "payload": {
            "values": [np.nan, 0, -3.5],
            "nested": {"missing": float("nan")},
        },
    }

    assert publisher.publish_many("events", [event], key_field="event_id") == 1

    message = fake_producer.messages[0]
    assert message["key"] == "event-1"
    assert "NaN" not in message["value"]
    assert json.loads(message["value"]) == {
        "event_id": "event-1",
        "payload": {
            "nested": {"missing": None},
            "values": [None, 0, -3.5],
        },
        "score": None,
    }


def test_publisher_fails_if_sanitization_is_bypassed(
    fake_producer: FakeProducer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = KafkaPredictionPublisher(bootstrap_servers="kafka:9092")
    monkeypatch.setattr(kafka_module, "sanitize_json", lambda value: value)

    with pytest.raises(ValueError, match="Out of range float values"):
        publisher.publish_many("events", [{"score": float("nan")}])

    assert fake_producer.messages == []
