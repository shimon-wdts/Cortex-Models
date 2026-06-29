from __future__ import annotations

import json
from typing import Any

from app.models.pipeline_contracts import InsightEvent


class KafkaPredictionPublisher:
    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str = "cortex-models",
        acks: str = "all",
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        from confluent_kafka import Producer
        if not bootstrap_servers:
            raise ValueError("Kafka bootstrap servers are not configured")
        config = {
            "bootstrap.servers": bootstrap_servers,
            "client.id": client_id,
            "acks": acks,
            **(extra_config or {}),
        }
        self._producer = Producer(config)

    def publish_many(self, topic: str, events: list[dict[str, Any]], key_field: str | None = None) -> int:
        delivery_errors: list[Exception] = []

        def delivery_callback(error, _message) -> None:
            if error is not None:
                delivery_errors.append(RuntimeError(str(error)))

        for event in events:
            # payload = event.model_dump(mode="json")
            key = _event_key(event, key_field)
            self._producer.produce(
                topic=topic,
                key=key,
                value=json.dumps(event, separators=(",", ":"), sort_keys=True),
                callback=delivery_callback,
            )
            self._producer.poll(0)

        self._producer.flush()
        if delivery_errors:
            raise delivery_errors[0]
        return len(events)


def _event_key(payload: dict[str, Any], key_field: str | None) -> str | None:
    if not key_field:
        return None
    return payload.get(key_field)
    # for entity in payload.get("entity", []):
    #     if entity.get("type") == key_field:
    #         return str(entity.get("id"))
    # value = payload.get(key_field)
    # return str(value) if value is not None else None
