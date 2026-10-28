from __future__ import annotations

from typing import Any, Protocol

from app.models.pipeline_contracts import ModelStoreConfig, ModelStoreConnectionConfig


class PredictiveModel(Protocol):
    def predict(self, data: Any) -> Any:
        ...


class ModelLoader(Protocol):
    def load(
        self,
        connection: ModelStoreConnectionConfig,
        config: ModelStoreConfig,
    ) -> PredictiveModel:
        ...


class MLflowModelLoader:
    def load(
        self,
        connection: ModelStoreConnectionConfig,
        config: ModelStoreConfig,
    ) -> PredictiveModel:
        import mlflow

        if connection.tracking_uri:
            mlflow.set_tracking_uri(connection.tracking_uri)

        return mlflow.pyfunc.load_model(config.uri)


def load_model(
    connection: ModelStoreConnectionConfig,
    config: ModelStoreConfig,
) -> PredictiveModel:
    if connection.provider != "mlflow":
        raise ValueError(f"Unsupported model store provider: {connection.provider}")
    return MLflowModelLoader().load(connection, config)
