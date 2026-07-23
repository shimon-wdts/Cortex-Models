from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4
from venv import logger

import pandas as pd
from prefect.logging import get_run_logger

from app.clients.kafka import KafkaPredictionPublisher
from app.clients.model_store import load_model
from app.clients.postgres import PostgresQueryClient
from app.metrics import MODEL_PREDICTIONS_PUBLISHED_TOTAL
from app.models.pipeline_contracts import (
    ExecutionMode,
    InsightEvent,
    PipelineStep,
    RunContext,
    StepResult,
)
from app.pipelines import get_pipeline
from app.services.imports import import_callable
from app.services.model_registry import ModelRegistry, get_model_registry
from app.services.observability import observe_step, record_rows


def create_run_context(
    model_name: str,
    execution_mode: ExecutionMode,
    run_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> RunContext:
    input_parameters = parameters or {}
    return RunContext(
        model_name=model_name,
        run_id=run_id or _new_run_id(),
        execution_mode=execution_mode,
        parameters=input_parameters,
        query_parameters=get_pipeline(model_name).get_query_params(input_parameters),
        model_store_dir=get_model_registry().get_model_store_dir(model_name)
    )


def extract_data(
    context: RunContext,
    registry: ModelRegistry | None = None,
    client: PostgresQueryClient | None = None,
) -> dict[str, pd.DataFrame]:
    registry = registry or get_model_registry()
    config = registry.get_model(context.model_name)
    with observe_step(context, PipelineStep.EXTRACT):
        if context.query_parameters.get("_skip_extract"):
            logger = get_run_logger()
            logger.info(PipelineStep.EXTRACT + " skipped by pipeline parameters")
            record_rows(context, PipelineStep.EXTRACT, 0)
            return {}

        client = client or PostgresQueryClient(registry.postgres.get("replica_url", ""))
        data: dict[str, pd.DataFrame] = {}
        rows = 0
        for query in config.queries:
            frame = client.query(query.sql, _render_params(query.params, context.query_parameters))
            if frame.empty:
                frame = pd.DataFrame(columns=query.df_columns)
            data[query.name] = frame
            rows += len(frame)
        logger = get_run_logger()
        logger.info(PipelineStep.EXTRACT + " total rows: " + str(rows))
        record_rows(context, PipelineStep.EXTRACT, rows)
        return data


def generate_features(
    context: RunContext,
    raw_data: dict[str, Any] | list[dict[str, Any]],
    registry: ModelRegistry | None = None,
) -> Any:
    registry = registry or get_model_registry()
    config = registry.get_model(context.model_name)
    with observe_step(context, PipelineStep.FEATURES):
        raw_frames = _coerce_raw_data_by_query(raw_data)
        features = get_pipeline(context.model_name).build_feature(raw_frames, context.query_parameters)
        logger = get_run_logger()
        logger.info(PipelineStep.FEATURES + " total rows: " + str(len(features)))
        record_rows(context, PipelineStep.FEATURES, len(features))
        return features


def run_inference(
    context: RunContext,
    features: Any,
    registry: ModelRegistry | None = None,
) -> list[dict[str, Any]]:
    registry = registry or get_model_registry()
    config = registry.get_model(context.model_name)
    model_dir = registry.get_model_store_dir(context.model_name)
    with observe_step(context, PipelineStep.INFERENCE):
        logger = get_run_logger()
        predictions = get_pipeline(context.model_name).run_inference(features, context)
        logger.info(PipelineStep.INFERENCE + " total rows: " + str(len(predictions)))
        record_rows(context, PipelineStep.INFERENCE, len(predictions))
        return predictions


def build_prediction_events(
    context: RunContext,
    prediction_records: list[dict[str, Any]],
    feature_records: list[dict[str, Any]],
    registry: ModelRegistry | None = None,
) -> list[InsightEvent]:
    registry = registry or get_model_registry()
    config = registry.get_model(context.model_name)
    adapter = import_callable(config.output.adapter)
    events = adapter(
        config,
        pd.DataFrame(prediction_records),
        pd.DataFrame(feature_records),
        context,
    )
    if not all(isinstance(event, InsightEvent) for event in events):
        events = [InsightEvent.model_validate(event) for event in events]
    return events


def publish_predictions(
    context: RunContext,
    prediction_records: list[dict[str, Any]],
    registry: ModelRegistry | None = None,
) -> StepResult:
    registry = registry or get_model_registry()
    config = registry.get_model(context.model_name)
    with observe_step(context, PipelineStep.PUBLISH):
        # events = build_prediction_events(context, prediction_records, registry)
        logger.info(PipelineStep.PUBLISH + " total rows: " + str(len(prediction_records)))
        kafka_config = registry.kafka
        publisher = KafkaPredictionPublisher(
            bootstrap_servers=kafka_config.get("bootstrap_servers", ""),
            client_id=kafka_config.get("client_id", "cortex-models"),
            acks=kafka_config.get("acks", "all"),
            security_protocol=kafka_config.get("security_protocol"),
            sasl_mechanism=kafka_config.get("sasl_mechanism"),
            sasl_username=kafka_config.get("sasl_username"),
            sasl_password=kafka_config.get("sasl_password"),
            ssl_ca_location=kafka_config.get("ssl_ca_location"),
            ssl_endpoint_identification_algorithm=kafka_config.get("ssl_endpoint_identification_algorithm"),
            extra_config=kafka_config.get("producer_config"),
        )
        published = publisher.publish_many(
            topic=config.kafka.topic,
            events=prediction_records,
            key_field=config.kafka.key_field,
        )
        MODEL_PREDICTIONS_PUBLISHED_TOTAL.labels(
            context.model_name,
            context.run_id,
            config.kafka.topic,
        ).inc(published)
        record_rows(context, PipelineStep.PUBLISH, published)
        return StepResult(
            model_name=context.model_name,
            run_id=context.run_id,
            step=PipelineStep.PUBLISH,
            status="completed",
            rows=len(prediction_records),
            published=published,
            topic=config.kafka.topic,
        )

def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{uuid4().hex[:16]}"


def _render_params(params: dict[str, Any], runtime_params: dict[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            runtime_key = value.strip("{} ")
            rendered[key] = runtime_params.get(runtime_key)
        else:
            rendered[key] = value
    return rendered


def _coerce_raw_data_by_query(raw_data: dict[str, Any] | list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    if isinstance(raw_data, dict):
        return {
            name: value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            for name, value in raw_data.items()
        }
    return {"default": pd.DataFrame(raw_data)}
