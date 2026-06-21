from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

from app.metrics import MODEL_STEP_DURATION_SECONDS, MODEL_STEP_ROWS_TOTAL, MODEL_STEP_RUNS_TOTAL
from app.models.pipeline_contracts import PipelineStep, RunContext


logger = logging.getLogger(__name__)


def log_step_event(
    event: str,
    context: RunContext,
    step: PipelineStep,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "model_name": context.model_name,
        "run_id": context.run_id,
        "execution_mode": context.execution_mode.value,
        "step": step.value,
        **fields,
    }
    logger.info(json.dumps(payload, default=str, sort_keys=True))


@contextmanager
def observe_step(context: RunContext, step: PipelineStep) -> Iterator[None]:
    started_at = time.perf_counter()
    log_step_event("model_step_started", context, step)
    try:
        yield
    except Exception as exc:
        duration = time.perf_counter() - started_at
        MODEL_STEP_RUNS_TOTAL.labels(context.model_name, context.run_id, step.value, "failed").inc()
        MODEL_STEP_DURATION_SECONDS.labels(context.model_name, context.run_id, step.value, "failed").observe(duration)
        log_step_event(
            "model_step_failed",
            context,
            step,
            duration_seconds=duration,
            failure_type=type(exc).__name__,
            failure_reason=str(exc),
        )
        raise
    else:
        duration = time.perf_counter() - started_at
        MODEL_STEP_RUNS_TOTAL.labels(context.model_name, context.run_id, step.value, "completed").inc()
        MODEL_STEP_DURATION_SECONDS.labels(
            context.model_name, context.run_id, step.value, "completed"
        ).observe(duration)
        log_step_event("model_step_completed", context, step, duration_seconds=duration)


def record_rows(context: RunContext, step: PipelineStep, rows: int) -> None:
    MODEL_STEP_ROWS_TOTAL.labels(context.model_name, context.run_id, step.value).inc(rows)
    log_step_event("model_step_rows", context, step, rows=rows)
