from __future__ import annotations

from app.pipelines.base import Pipeline
from app.pipelines.lucky6_bigtiger import Lucky6BigTigerPipeline
from app.pipelines.predicted_fills import PredictiveFillsPipeline


_DEFAULT_PIPELINE = Pipeline()
_PIPELINES: dict[str, Pipeline] = {
    "ShoeAdvantage": Lucky6BigTigerPipeline(),
    "predicted_fills": PredictiveFillsPipeline(),
}


def get_pipeline(model_name: str) -> Pipeline:
    return _PIPELINES.get(model_name, _DEFAULT_PIPELINE)
