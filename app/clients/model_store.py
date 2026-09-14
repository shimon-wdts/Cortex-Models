from __future__ import annotations

from typing import Any, Protocol

from app.models.pipeline_contracts import ModelStoreConfig
from app.services.imports import import_callable


class PredictiveModel(Protocol):
    def predict(self, data: Any) -> Any:
        ...


class ModelLoader(Protocol):
    def load(self, config: ModelStoreConfig) -> PredictiveModel:
        ...


class MLflowModelLoader:
    def load(self, config: ModelStoreConfig) -> PredictiveModel:
        import mlflow

        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)

        model_uri = config.uri
        if not model_uri:
            if not config.registered_model_name:
                raise ValueError("MLflow model config requires registered_model_name or uri")
            model_uri = f"models:/{config.registered_model_name}/{config.version}"

        return mlflow.pyfunc.load_model(model_uri)


def load_model(config: ModelStoreConfig) -> PredictiveModel:
    if config.provider == "custom":
        if not config.loader:
            raise ValueError("Custom model store config requires loader")
        loader_factory = import_callable(config.loader)
        loader = loader_factory()
        return loader.load(config)
    return MLflowModelLoader().load(config)
