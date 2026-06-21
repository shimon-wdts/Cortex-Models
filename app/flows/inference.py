from __future__ import annotations

from typing import Any

from prefect import flow, task

from app.models.pipeline_contracts import ExecutionMode, PipelineStep, RunContext, StepResult
from app.services.pipeline import (
    create_run_context,
    extract_data,
    generate_features,
    publish_predictions,
    run_inference,
    run_step,
)


@task(name="extract-data", retries=2, retry_delay_seconds=30)
def extract_data_task(context: RunContext) -> list[dict[str, Any]]:
    return extract_data(context)


@task(name="feature-engineering", retries=1, retry_delay_seconds=15)
def feature_engineering_task(
    context: RunContext,
    raw_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return generate_features(context, raw_records)


@task(name="inference", retries=1, retry_delay_seconds=15)
def inference_task(
    context: RunContext,
    feature_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return run_inference(context, feature_records)


@task(name="publish-predictions", retries=3, retry_delay_seconds=20)
def publish_predictions_task(
    context: RunContext,
    prediction_records: list[dict[str, Any]],
    feature_records: list[dict[str, Any]],
) -> StepResult:
    return publish_predictions(context, prediction_records, feature_records)


@task(name="single-step", retries=1, retry_delay_seconds=15)
def single_step_task(
    context: RunContext,
    step: PipelineStep,
    inputs: dict[str, Any],
) -> StepResult:
    return run_step(context, step, inputs)


@flow(name="cortex-model-pipeline")
def full_pipeline_flow(
    model_name: str,
    run_id: str | None = None,
    parameters: dict[str, Any] | None = None,
    execution_mode: str = ExecutionMode.SCHEDULED.value,
) -> dict[str, Any]:
    context = create_run_context(
        model_name=model_name,
        execution_mode=ExecutionMode(execution_mode),
        run_id=run_id,
        parameters=parameters,
    )
    raw_records = extract_data_task(context)
    feature_records = feature_engineering_task(context, raw_records)
    prediction_records = inference_task(context, feature_records)
    result = publish_predictions_task(context, prediction_records, feature_records)
    return result.model_dump(mode="json")


@flow(name="cortex-model-step")
def step_flow(
    model_name: str,
    step: str,
    run_id: str | None = None,
    parameters: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    execution_mode: str = ExecutionMode.MANUAL.value,
) -> dict[str, Any]:
    context = create_run_context(
        model_name=model_name,
        execution_mode=ExecutionMode(execution_mode),
        run_id=run_id,
        parameters=parameters,
    )
    result = single_step_task(context, PipelineStep(step), inputs or {})
    return result.model_dump(mode="json")
