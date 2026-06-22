from __future__ import annotations

import subprocess
from datetime import timedelta
from typing import Any

from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule, RRuleSchedule
from prefect.deployments.runner import EntrypointType

from app.flows.inference import full_pipeline_flow
from app.models.pipeline_contracts import ExecutionMode, ScheduleConfig
from app.services.model_registry import get_model_registry


def deploy_configured_models() -> list[str]:
    registry = get_model_registry()
    deployment_ids: list[str] = []
    prefect_config = registry.prefect
    work_pool_name = prefect_config.get("work_pool_name")
    work_pool_type = prefect_config.get("work_pool_type", "process")
    work_queue_name = prefect_config.get("work_queue_name")
    if work_pool_name:
        _ensure_work_pool(work_pool_name, work_pool_type)

    for model_name, config in registry.list_models().items():
        deployment_id = full_pipeline_flow.deploy(
            name=model_name,
            work_pool_name=work_pool_name,
            work_queue_name=work_queue_name,
            build=False,
            push=False,
            schedule=_prefect_schedule(config.schedule) if config.schedule.enabled else None,
            paused=False,
            concurrency_limit=config.schedule.concurrency_limit,
            parameters={
                "model_name": model_name,
                "execution_mode": ExecutionMode.SCHEDULED.value,
                "parameters": config.parameters
            },
            tags=["cortex-models", model_name, "batch"],
            description=config.description,
            version=config.version,
            entrypoint_type=EntrypointType.MODULE_PATH,
            ignore_warnings=True,
        )
        deployment_ids.append(str(deployment_id))

        # for step in PipelineStep:
        #     step_deployment_id = step_flow.deploy(
        #         name=f"{model_name}-{step.value}",
        #         work_pool_name=work_pool_name,
        #         work_queue_name=work_queue_name,
        #         build=False,
        #         push=False,
        #         parameters={
        #             "model_name": model_name,
        #             "step": step.value,
        #             "execution_mode": ExecutionMode.MANUAL.value,
        #         },
        #         paused=False,
        #         tags=["cortex-models", model_name, "manual-step", step.value],
        #         description=f"Manual {step.value} flow for {model_name}",
        #         entrypoint_type=EntrypointType.MODULE_PATH,
        #         ignore_warnings=True,
        #     )
        #     deployment_ids.append(str(step_deployment_id))

    return deployment_ids


def _ensure_work_pool(work_pool_name: str, work_pool_type: str) -> None:
    inspect_result = subprocess.run(
        ["prefect", "work-pool", "inspect", work_pool_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspect_result.returncode == 0:
        return

    subprocess.run(
        [
            "prefect",
            "work-pool",
            "create",
            work_pool_name,
            "--type",
            work_pool_type,
            "--no-prompt",
        ],
        check=True,
    )


def _prefect_schedule(config: ScheduleConfig) -> Any:
    if config.type == "cron":
        if not config.cron:
            raise ValueError("Cron schedule requires cron")
        return CronSchedule(cron=config.cron, timezone=config.timezone)
    if config.type == "interval":
        if not config.interval:
            raise ValueError("Interval schedule requires interval")
        return IntervalSchedule(interval=timedelta(seconds=float(config.interval)), timezone=config.timezone)
    if not config.rrule:
        raise ValueError("RRule schedule requires rrule")
    return RRuleSchedule(rrule=config.rrule, timezone=config.timezone)


if __name__ == "__main__":
    for created_deployment_id in deploy_configured_models():
        print(created_deployment_id)
