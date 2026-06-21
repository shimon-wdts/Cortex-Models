from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

import pandas as pd

from app.models.pipeline_contracts import EntityReference, InsightEvent, ModelPipelineConfig, ModelTrace, RunContext


def default_predict_adapter(model: Any, features: pd.DataFrame, _params: dict[str, Any] | None = None) -> pd.DataFrame:
    predictions = model.predict(features)
    if isinstance(predictions, pd.DataFrame):
        return predictions
    if isinstance(predictions, pd.Series):
        return predictions.to_frame(name="prediction")
    if isinstance(predictions, list):
        return pd.DataFrame(predictions)
    return pd.DataFrame({"prediction": predictions})


def generic_insight_event_adapter(
    config: ModelPipelineConfig,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    context: RunContext,
) -> list[InsightEvent]:
    if len(predictions) != len(features):
        raise ValueError("Prediction row count must match feature row count")

    events: list[InsightEvent] = []
    now = datetime.now(UTC)
    for index, feature_row in features.reset_index(drop=True).iterrows():
        prediction_row = predictions.reset_index(drop=True).iloc[index].to_dict()
        feature_values = feature_row.to_dict()
        events.append(
            InsightEvent(
                insights_id=f"evt_{uuid4().hex[:17].upper()}",
                occurred_at=now,
                gaming_day=_gaming_day(feature_values),
                source=config.source,
                env=config.env,
                version=config.output_schema_version,
                model=ModelTrace(
                    type=config.type,
                    version=config.model_store.version,
                    feature_version=config.features.version,
                    run_id=context.run_id,
                ),
                entity=_entities(config, feature_values),
                severity=str(prediction_row.get("severity") or config.severity_default),
                payload={
                    "result": _result_payload(prediction_row),
                    "presentation": _presentation_payload(prediction_row),
                    "actions": config.output.actions,
                },
            )
        )
    return events


def _gaming_day(row: dict[str, Any]) -> date:
    value = row.get("gaming_day")
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return date.fromisoformat(value[:10])
    return datetime.now(UTC).date()


def _entities(config: ModelPipelineConfig, row: dict[str, Any]) -> list[EntityReference]:
    entities: list[EntityReference] = []
    for mapping in config.output.entity_mappings:
        value = row.get(mapping.field)
        if value is None:
            continue
        entities.append(
            EntityReference(
                type=mapping.type,
                id=str(value),
                present_in_user_interface=mapping.present_in_user_interface,
            )
        )
    return entities


def _result_payload(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_class": prediction.get("decision_class", prediction.get("prediction", "unknown")),
        "score": prediction.get("score"),
        "confidence": prediction.get("confidence", "unknown"),
        "modeled_impact": prediction.get("modeled_impact", {}),
        "expected_deficit": prediction.get("expected_deficit"),
    }


def _presentation_payload(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "headline": prediction.get("headline", "Model insight"),
        "trigger_metric": prediction.get("trigger_metric", "model_score"),
        "recommendations": prediction.get("recommendations", []),
        "chart": prediction.get("chart", {}),
    }
