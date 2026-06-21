from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineStep(StrEnum):
    EXTRACT = "extract_data"
    FEATURES = "feature_engineering"
    INFERENCE = "inference"
    PUBLISH = "publish"


class ExecutionMode(StrEnum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class ScheduleConfig(BaseModel):
    enabled: bool = False
    type: Literal["cron", "interval", "rrule"] = "cron"
    cron: str | None = None
    interval: int | float | None = None
    rrule: str | None = None
    timezone: str = "UTC"
    concurrency_limit: int | None = 1


class QueryConfig(BaseModel):
    name: str
    sql: str
    params: dict[str, Any] = Field(default_factory=dict)
    df_columns: list[str] = Field(default_factory=list)


class FeatureConfig(BaseModel):
    version: str
    builder: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelStoreConfig(BaseModel):
    provider: Literal["mlflow", "custom"] = "mlflow"
    tracking_uri: str | None = None
    registered_model_name: str | None = None
    version: str
    uri: str | None = None
    loader: str | None = None


class InferenceConfig(BaseModel):
    adapter: str
    params: dict[str, Any] = Field(default_factory=dict)


class EntityMappingConfig(BaseModel):
    type: str
    field: str
    present_in_user_interface: bool = False


class OutputConfig(BaseModel):
    adapter: str
    entity_mappings: list[EntityMappingConfig] = Field(default_factory=list)
    actions: dict[str, Any] = Field(default_factory=dict)


class KafkaTopicConfig(BaseModel):
    topic: str
    key_field: str | None = None


class ModelPipelineConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    description: str = ""
    version: str = "1.0.0"
    type: str
    source: str
    env: str = "local"
    output_schema_version: float = 1.0
    severity_default: str = "medium"
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    queries: list[QueryConfig] = Field(default_factory=list)
    features: FeatureConfig
    model_store: ModelStoreConfig
    inference: InferenceConfig
    output: OutputConfig
    kafka: KafkaTopicConfig


class ModelRegistryConfig(BaseModel):
    prefect: dict[str, Any] = Field(default_factory=dict)
    postgres: dict[str, Any] = Field(default_factory=dict)
    kafka: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, ModelPipelineConfig] = Field(default_factory=dict)


class RunContext(BaseModel):
    model_name: str
    run_id: str
    execution_mode: ExecutionMode
    parameters: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    model_name: str
    run_id: str
    step: PipelineStep
    status: Literal["completed", "failed"]
    rows: int = 0
    published: int = 0
    topic: str | None = None
    error: str | None = None


class ModelTrace(BaseModel):
    type: str
    version: str
    feature_version: str
    run_id: str


class EntityReference(BaseModel):
    type: str
    id: str
    present_in_user_interface: bool


class InsightEvent(BaseModel):
    insights_id: str
    occurred_at: datetime
    gaming_day: date
    source: str
    env: str
    version: float
    model: ModelTrace
    entity: list[EntityReference]
    severity: str
    payload: dict[str, Any]

    @field_validator("occurred_at")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
