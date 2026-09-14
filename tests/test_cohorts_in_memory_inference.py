from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.models.pipeline_contracts import ExecutionMode, RunContext
from app.pipelines.cohorts.cohort_inference import infer_cohorts_from_frame
from app.pipelines.cohorts.pipeline import CohortsPipeline
from app.pipelines.cohorts.recommendation_inference import infer_recommendations_from_frame
from app.pipelines.cohorts.source import CohortsFeatureResult
from app.pipelines.cohorts.utils import assign_period


MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "models_store" / "cohorts"


def test_query_params_cover_eligible_players_in_uncapped_rolling_window() -> None:
    params = CohortsPipeline("cohort").get_query_params({"gaming_day_end": "2026-09-06"})

    assert str(params["gaming_day_start"]) == "2026-08-17"
    assert str(params["gaming_day_end_exclusive"]) == "2026-09-07"
    assert params["observation_days"] == 21
    assert params["minimum_bets"] == 30
    assert params["extract_batch_size"] == 500
    assert params["extract_workers"] == 4
    assert params["json_limit"] == 0


def test_rolling_period_assignment_crosses_month_boundary() -> None:
    days = pd.Series(pd.to_datetime(["2026-08-17", "2026-08-23", "2026-08-24", "2026-08-30", "2026-08-31", "2026-09-06"]))

    assert assign_period(days, start_day="2026-08-17").tolist() == [
        "Week 1",
        "Week 1",
        "Week 2",
        "Week 2",
        "Week 3",
        "Week 3",
    ]


def test_frame_inference_keeps_intermediate_results_in_memory() -> None:
    features = pd.read_csv(
        MODEL_DIR / "samples" / "input" / "player_period_features_sample.csv",
        low_memory=False,
    )
    original = features.copy(deep=True)

    cohorts = infer_cohorts_from_frame(features, MODEL_DIR / "cohort_model.json")
    recommendations = infer_recommendations_from_frame(
        cohorts,
        MODEL_DIR / "recommendation_model_metadata.json",
    )

    pd.testing.assert_frame_equal(features, original)
    assert len(cohorts) == len(features)
    assert len(recommendations) == features["player_id"].nunique()
    assert "cohort_id" in cohorts.columns
    assert "recommended_path" in recommendations.columns


def test_pipeline_passes_dataframes_between_inference_stages(monkeypatch, tmp_path: Path) -> None:
    period_features = pd.DataFrame(
        [{"player_id": "1", "period": "Total", "bet_count": 30, "eligible": True}]
    )
    total_features = period_features.copy()
    features = CohortsFeatureResult(period_features, total_features)
    cohorts = pd.DataFrame([{"player_id": "1", "period": "Total", "cohort_id": 2}])
    recommendations = pd.DataFrame([{"player_id": "1", "recommended_path": "Maintain"}])
    pipeline = CohortsPipeline("cohort")

    def fake_infer_cohorts(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), period_features)
        assert model_path == tmp_path / "cohort_model.json"
        return cohorts

    def fake_infer_recommendations(
        frame: pd.DataFrame,
        metadata_path: Path,
        recommend_all: bool = True,
    ) -> pd.DataFrame:
        assert frame is cohorts
        assert metadata_path == tmp_path / "recommendation_model_metadata.json"
        assert recommend_all is True
        return recommendations

    def fake_build_events(
        feature_result: CohortsFeatureResult,
        cohort_frame: pd.DataFrame,
        recommendation_frame: pd.DataFrame,
    ) -> list[dict[str, str]]:
        pd.testing.assert_frame_equal(feature_result.player_period_features.reset_index(drop=True), period_features)
        pd.testing.assert_frame_equal(feature_result.player_total_features.reset_index(drop=True), total_features)
        assert feature_result.metadata["minimum_bets"] == 30
        assert cohort_frame is cohorts
        assert recommendation_frame is recommendations
        return [{"player_id": "1"}]

    monkeypatch.setattr("app.pipelines.cohorts.pipeline.infer_cohorts_from_frame", fake_infer_cohorts)
    monkeypatch.setattr(
        "app.pipelines.cohorts.pipeline.infer_recommendations_from_frame",
        fake_infer_recommendations,
    )
    monkeypatch.setattr(pipeline, "_build_events", fake_build_events)
    context = RunContext(
        model_name="cohort",
        run_id="test-run",
        execution_mode=ExecutionMode.MANUAL,
        query_parameters={"recommend_all": True, "json_limit": 0},
        model_store_dir=str(tmp_path),
    )

    assert pipeline.run_inference(features, context) == [{"player_id": "1"}]


def test_pipeline_only_scores_players_with_minimum_history(monkeypatch, tmp_path: Path) -> None:
    period_features = pd.DataFrame(
        [
            {"player_id": "eligible", "period": "Week 1", "bet_count": 15, "eligible": False},
            {"player_id": "eligible", "period": "Total", "bet_count": 30, "eligible": True},
            {"player_id": "sparse", "period": "Week 1", "bet_count": 10, "eligible": False},
            {"player_id": "sparse", "period": "Total", "bet_count": 20, "eligible": False},
        ]
    )
    features = CohortsFeatureResult(
        player_period_features=period_features,
        player_total_features=period_features[period_features["period"].eq("Total")].copy(),
    )
    pipeline = CohortsPipeline("playerscore")

    def fake_infer_cohorts(frame: pd.DataFrame, _model_path: Path) -> pd.DataFrame:
        assert set(frame["player_id"]) == {"eligible"}
        return frame

    def fake_infer_recommendations(
        frame: pd.DataFrame,
        _metadata_path: Path,
        recommend_all: bool = True,
    ) -> pd.DataFrame:
        assert set(frame["player_id"]) == {"eligible"}
        return frame[frame["period"].eq("Total")].copy()

    def fake_build_events(
        feature_result: CohortsFeatureResult,
        _cohorts: pd.DataFrame,
        _recommendations: pd.DataFrame,
    ) -> list[dict[str, str]]:
        assert feature_result.metadata["minimum_bets"] == 30
        assert set(feature_result.player_total_features["player_id"]) == {"eligible"}
        return [{"player_id": "eligible"}]

    monkeypatch.setattr("app.pipelines.cohorts.pipeline.infer_cohorts_from_frame", fake_infer_cohorts)
    monkeypatch.setattr(
        "app.pipelines.cohorts.pipeline.infer_recommendations_from_frame",
        fake_infer_recommendations,
    )
    monkeypatch.setattr(pipeline, "_build_events", fake_build_events)
    context = RunContext(
        model_name="playerscore",
        run_id="test-run",
        execution_mode=ExecutionMode.MANUAL,
        query_parameters={"recommend_all": True, "json_limit": 0},
        model_store_dir=str(tmp_path),
    )

    assert pipeline.run_inference(features, context) == [{"player_id": "eligible"}]
