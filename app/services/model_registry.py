from __future__ import annotations

from typing import Any

from dynaconf import Dynaconf

from app.core.config import settings
from app.models.pipeline_contracts import ModelPipelineConfig, ModelRegistryConfig


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return dict(value)


class ModelRegistry:
    def __init__(self, dynaconf_settings: Dynaconf = settings) -> None:
        raw_registry = _as_plain_dict(dynaconf_settings.get("model_registry", {}))
        self._registry = ModelRegistryConfig.model_validate(raw_registry)

    @property
    def prefect(self) -> dict[str, Any]:
        return self._registry.prefect

    @property
    def postgres(self) -> dict[str, Any]:
        return self._registry.postgres

    @property
    def kafka(self) -> dict[str, Any]:
        return self._registry.kafka

    def list_models(self, include_disabled: bool = False) -> dict[str, ModelPipelineConfig]:
        if include_disabled:
            return self._registry.models
        return {
            name: config
            for name, config in self._registry.models.items()
            if config.enabled
        }

    def get_model(self, model_name: str, include_disabled: bool = False) -> ModelPipelineConfig:
        models = self.list_models(include_disabled=include_disabled)
        try:
            return models[model_name]
        except KeyError as exc:
            raise ModelNotRegisteredError(model_name) from exc

    def validate_model_registered(self, model_name: str) -> None:
        self.get_model(model_name)

    def full_deployment_name(self, model_name: str) -> str:
        return f"cortex-model-pipeline/{model_name}"

    def step_deployment_name(self, model_name: str, step: str) -> str:
        return f"cortex-model-step/{model_name}-{step}"


def get_model_registry() -> ModelRegistry:
    return ModelRegistry(settings)


class ModelNotRegisteredError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model is not registered or is disabled: {model_name}")
