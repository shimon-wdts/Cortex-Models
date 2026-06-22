from __future__ import annotations

from app.pipelines.base import Pipeline
from app.pipelines.predicted_fills import PredictiveFillsPipeline


_DEFAULT_PIPELINE = Pipeline()
_PIPELINES: dict[str, Pipeline] = {
    "predicted_fills": PredictiveFillsPipeline(),
}


def get_pipeline(model_name: str) -> Pipeline:
    return _PIPELINES.get(model_name, _DEFAULT_PIPELINE)
