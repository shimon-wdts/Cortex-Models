from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pandas as pd

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
from app.services.imports import import_callable
from app.services.model_registry import ModelRegistry, get_model_registry
from app.services.observability import observe_step, record_rows


def create_run_context(
    model_name: str,
    execution_mode: ExecutionMode,
    run_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> RunContext:
    return RunContext(
        model_name=model_name,
        run_id=run_id or _new_run_id(),
        execution_mode=execution_mode,
        parameters=parameters or {},
    )


def extract_data(
    context: RunContext,
    registry: ModelRegistry | None = None,
) -> dict[str, pd.DataFrame]:
    registry = registry or get_model_registry()
    config = registry.get_model(context.model_name)
    with observe_step(context, PipelineStep.EXTRACT):
        client = PostgresQueryClient(registry.postgres.get("replica_url", ""))
        data: dict[str, pd.DataFrame] = {}
        rows = 0
        for query in config.queries:
            frame = client.query(query.sql, _render_params(query.params, context.parameters))
            if frame.empty:
                frame = pd.DataFrame(columns=query.df_columns)
            data[query.name] = frame
            rows += len(frame)
        record_rows(context, PipelineStep.EXTRACT, rows)
        return data


def generate_features(
    context: RunContext,
    raw_data: dict[str, Any] | list[dict[str, Any]],
    registry: ModelRegistry | None = None,
) -> list[dict[str, Any]]:
    registry = registry or get_model_registry()
    config = registry.get_model(context.model_name)
    with observe_step(context, PipelineStep.FEATURES):
        raw_frames = _coerce_raw_data_by_query(raw_data)
        builder = import_callable(config.features.builder)
        features = builder(raw_frames, config.features.params)
        if not isinstance(features, pd.DataFrame):
            raise TypeError("Feature builder must return a pandas DataFrame")
        record_rows(context, PipelineStep.FEATURES, len(features))
        return features.to_dict(orient="records")


def run_inference(
    context: RunContext,
    feature_records: list[dict[str, Any]],
    registry: ModelRegistry | None = None,
) -> list[dict[str, Any]]:
    registry = registry or get_model_registry()
    config = registry.get_model(context.model_name)
    with observe_step(context, PipelineStep.INFERENCE):
        features = pd.DataFrame(feature_records)
        model = load_model(config.model_store)
        adapter = import_callable(config.inference.adapter)
        predictions = adapter(model, features, config.inference.params)
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("Inference adapter must return a pandas DataFrame")
        record_rows(context, PipelineStep.INFERENCE, len(predictions))
        return predictions.to_dict(orient="records")


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
    feature_records: list[dict[str, Any]],
    registry: ModelRegistry | None = None,
) -> StepResult:
    registry = registry or get_model_registry()
    config = registry.get_model(context.model_name)
    with observe_step(context, PipelineStep.PUBLISH):
        events = build_prediction_events(context, prediction_records, feature_records, registry)
        kafka_config = registry.kafka
        publisher = KafkaPredictionPublisher(
            bootstrap_servers=kafka_config.get("bootstrap_servers", ""),
            client_id=kafka_config.get("client_id", "cortex-models"),
            acks=kafka_config.get("acks", "all"),
            extra_config=kafka_config.get("producer_config"),
        )
        published = publisher.publish_many(
            topic=config.kafka.topic,
            events=events,
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
            rows=len(events),
            published=published,
            topic=config.kafka.topic,
        )


def run_full_pipeline(context: RunContext, registry: ModelRegistry | None = None) -> StepResult:
    registry = registry or get_model_registry()
    raw_data = extract_data(context, registry)
    feature_records = generate_features(context, raw_data, registry)
    prediction_records = run_inference(context, feature_records, registry)
    return publish_predictions(context, prediction_records, feature_records, registry)


def run_step(
    context: RunContext,
    step: PipelineStep,
    inputs: dict[str, Any] | None = None,
    registry: ModelRegistry | None = None,
) -> StepResult:
    registry = registry or get_model_registry()
    inputs = inputs or {}

    if step == PipelineStep.EXTRACT:
        raw_data = extract_data(context, registry)
        return StepResult(
            model_name=context.model_name,
            run_id=context.run_id,
            step=step,
            status="completed",
            rows=sum(len(frame) for frame in raw_data.values()),
        )
    if step == PipelineStep.FEATURES:
        feature_records = generate_features(
            context,
            inputs.get("raw_data", inputs.get("raw_records", {})),
            registry,
        )
        return StepResult(
            model_name=context.model_name,
            run_id=context.run_id,
            step=step,
            status="completed",
            rows=len(feature_records),
        )
    if step == PipelineStep.INFERENCE:
        prediction_records = run_inference(context, inputs.get("feature_records", []), registry)
        return StepResult(
            model_name=context.model_name,
            run_id=context.run_id,
            step=step,
            status="completed",
            rows=len(prediction_records),
        )
    if step == PipelineStep.PUBLISH:
        return publish_predictions(
            context,
            inputs.get("prediction_records", []),
            inputs.get("feature_records", []),
            registry,
        )

    raise ValueError(f"Unsupported pipeline step: {step}")


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
            name: value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            for name, value in raw_data.items()
        }
    return {"default": pd.DataFrame(raw_data)}
