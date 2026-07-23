from __future__ import annotations

import json
from typing import Any

class KafkaPredictionPublisher:
    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str = "cortex-models",
        acks: str = "all",
        security_protocol: str | None = None,
        sasl_mechanism: str | None = None,
        sasl_username: str | None = None,
        sasl_password: str | None = None,
        ssl_ca_location: str | None = None,
        ssl_endpoint_identification_algorithm: str | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        from confluent_kafka import Producer

        if not bootstrap_servers:
            raise ValueError("Kafka bootstrap servers are not configured")
        config = {
            "bootstrap.servers": bootstrap_servers,
            "client.id": client_id,
            "acks": acks,
            **_configured_security_options(
                security_protocol=security_protocol,
                sasl_mechanism=sasl_mechanism,
                sasl_username=sasl_username,
                sasl_password=sasl_password,
                ssl_ca_location=ssl_ca_location,
                ssl_endpoint_identification_algorithm=ssl_endpoint_identification_algorithm,
            ),
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


def _configured_security_options(
    *,
    security_protocol: str | None,
    sasl_mechanism: str | None,
    sasl_username: str | None,
    sasl_password: str | None,
    ssl_ca_location: str | None,
    ssl_endpoint_identification_algorithm: str | None,
) -> dict[str, str]:
    options = {
        "security.protocol": security_protocol,
        "sasl.mechanism": sasl_mechanism,
        "sasl.username": sasl_username,
        "sasl.password": sasl_password,
        "ssl.ca.location": ssl_ca_location,
        "ssl.endpoint.identification.algorithm": ssl_endpoint_identification_algorithm,
    }
    return {key: value for key, value in options.items() if value}


def _event_key(payload: dict[str, Any], key_field: str | None) -> str | None:
    if not key_field:
        return None
    return payload.get(key_field)
    # for entity in payload.get("entity", []):
    #     if entity.get("type") == key_field:
    #         return str(entity.get("id"))
    # value = payload.get(key_field)
    # return str(value) if value is not None else None
