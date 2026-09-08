from __future__ import annotations

import datetime
from typing import Any

from prefect import flow, runtime, task
from prefect.logging import get_run_logger

from app.clients.postgres import PostgresQueryClient
from app.models.pipeline_contracts import ExecutionMode, RunContext, StepResult
from app.services.pipeline import (
    create_run_context,
    extract_data,
    generate_features,
    publish_predictions,
    run_inference,
)
from app.services.model_registry import get_model_registry

COHORT_MODEL_NAMES = frozenset({"cohort", "cohort_tier_lift", "playerscore"})


def generate_flow_name() -> str:
    date = datetime.datetime.now(datetime.timezone.utc)
    try:
        flow_parameters = runtime.flow_run.parameters or {}
    except Exception:
        flow_parameters = {}
    parameters = flow_parameters.get("parameters") or {}
    model_name = flow_parameters.get("model_name") or parameters.get("model_name", "model")
    execution_mode = flow_parameters.get("execution_mode", ExecutionMode.SCHEDULED.value)
    return f"{model_name}-{execution_mode}-{date:%Y-%m-%d_%H-%M-%S}"


@task(name="extract-data", retries=2, retry_delay_seconds=30)
def extract_data_task(context: RunContext) -> dict[str, Any]:
    registry = get_model_registry()
    postgres = registry.postgres
    client = PostgresQueryClient(
        postgres.get("replica_url", ""),
        pool_size=postgres.get("pool_size", 1),
        pool_pre_ping=postgres.get("pool_pre_ping", True),
    )
    return extract_data(context, registry=registry, client=client)


@task(name="extract-and-feature-engineering", retries=1, retry_delay_seconds=30)
def cohort_extract_and_feature_task(context: RunContext) -> Any:
    registry = get_model_registry()
    postgres = registry.postgres
    client = PostgresQueryClient(
        postgres.get("replica_url", ""),
        pool_size=postgres.get("pool_size", 1),
        pool_pre_ping=postgres.get("pool_pre_ping", True),
    )
    raw_data = extract_data(context, registry=registry, client=client)
    get_run_logger().info(
        "cohort_handoff status=skipped reason=raw_frames_retained_in_process"
    )
    return generate_features(context, raw_data, registry=registry)


@task(name="feature-engineering", retries=1, retry_delay_seconds=15)
def feature_engineering_task(
    context: RunContext,
    raw_data: dict[str, Any] | list[dict[str, Any]],
) -> Any:
    return generate_features(context, raw_data)


@task(name="inference", retries=1, retry_delay_seconds=15)
def inference_task(
    context: RunContext,
    feature_records: Any,
) -> list[dict[str, Any]]:
    return run_inference(context, feature_records)


@task(name="publish-predictions", retries=3, retry_delay_seconds=20)
def publish_predictions_task(
    context: RunContext,
    prediction_records: list[dict[str, Any]],
) -> StepResult:
    return publish_predictions(context, prediction_records)


def uses_in_process_cohort_features(model_name: str) -> bool:
    return model_name in COHORT_MODEL_NAMES

@flow(
    name="cortex-model-pipeline",
    flow_run_name=generate_flow_name,
)
def full_pipeline_flow(
    model_name: str | None = None,
    parameters: dict[str, Any] = {},
    run_id: str | None = None,
    execution_mode: str = ExecutionMode.SCHEDULED.value,
) -> dict[str, Any]:
    input_parameters = parameters or {}
    resolved_model_name = model_name or input_parameters.get("model_name")
    if not resolved_model_name:
        raise ValueError("model_name must be provided directly or in parameters")

    context = create_run_context(
        model_name=resolved_model_name,
        execution_mode=ExecutionMode(execution_mode),
        run_id=run_id,
        parameters=input_parameters,
    )
    if uses_in_process_cohort_features(resolved_model_name):
        feature_records = cohort_extract_and_feature_task(context)
    else:
        raw_data = extract_data_task(context)
        feature_records = feature_engineering_task(context, raw_data)
    prediction_records = inference_task(context, feature_records)
    result = publish_predictions_task(context, prediction_records)
    return result.model_dump(mode="json")

