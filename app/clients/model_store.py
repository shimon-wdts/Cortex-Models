from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.models.pipeline_contracts import ModelStoreConfig, ModelStoreConnectionConfig


class PredictiveModel(Protocol):
    def predict_proba(self, data: Any) -> Any:
        ...


@dataclass(frozen=True)
class LoadedModel:
    model: PredictiveModel
    metadata: dict[str, Any]


class ModelLoader(Protocol):
    def load(
        self,
        connection: ModelStoreConnectionConfig,
        config: ModelStoreConfig,
    ) -> LoadedModel:
        ...


class MLflowModelLoader:
    def load(
        self,
        connection: ModelStoreConnectionConfig,
        config: ModelStoreConfig,
    ) -> LoadedModel:
        import mlflow
        import mlflow.xgboost

        if connection.tracking_uri:
            mlflow.set_tracking_uri(connection.tracking_uri)

        model_info = mlflow.models.get_model_info(config.uri)
        metadata = dict(model_info.metadata or {})
        if not metadata:
            if not model_info.run_id:
                raise ValueError(f"MLflow model has no metadata or source run: {config.uri}")
            metadata_uri = f"runs:/{model_info.run_id}/metadata/model_metadata.json"
            try:
                metadata = mlflow.artifacts.load_dict(metadata_uri)
            except Exception as exc:
                raise ValueError(
                    f"MLflow model metadata was not found at {metadata_uri}"
                ) from exc

        return LoadedModel(
            model=mlflow.xgboost.load_model(config.uri),
            metadata=metadata,
        )


def load_model(
    connection: ModelStoreConnectionConfig,
    config: ModelStoreConfig,
) -> LoadedModel:
    if connection.provider != "mlflow":
        raise ValueError(f"Unsupported model store provider: {connection.provider}")
    return MLflowModelLoader().load(connection, config)
