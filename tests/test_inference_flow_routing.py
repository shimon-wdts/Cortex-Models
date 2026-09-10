from __future__ import annotations

from types import SimpleNamespace

from app.flows.inference import cohort_extract_and_feature_task, uses_in_process_cohort_features


def test_cohort_models_keep_raw_extracts_inside_one_prefect_task() -> None:
    assert uses_in_process_cohort_features("cohort")
    assert uses_in_process_cohort_features("cohort_tier_lift")
    assert uses_in_process_cohort_features("playerscore")


def test_other_models_keep_separate_extract_and_feature_tasks() -> None:
    assert not uses_in_process_cohort_features("predicted_fills")
    assert not uses_in_process_cohort_features("ShoeAdvantage")


def test_combined_task_passes_raw_frames_directly_to_feature_engineering(monkeypatch) -> None:
    registry = SimpleNamespace(
        postgres={
            "replica_url": "postgresql://example",
            "pool_size": 2,
            "pool_pre_ping": True,
        }
    )
    client = object()
    context = object()
    raw_data = {"bets": object()}
    features = object()
    calls: list[str] = []

    monkeypatch.setattr("app.flows.inference.get_model_registry", lambda: registry)

    def fake_client(*args, **kwargs):
        assert args == ("postgresql://example",)
        assert kwargs == {"pool_size": 2, "pool_pre_ping": True}
        return client

    def fake_extract(received_context, *, registry, client):
        assert received_context is context
        calls.append("extract")
        return raw_data

    def fake_generate(received_context, received_raw_data, *, registry):
        assert received_context is context
        assert received_raw_data is raw_data
        calls.append("features")
        return features

    monkeypatch.setattr("app.flows.inference.PostgresQueryClient", fake_client)
    monkeypatch.setattr("app.flows.inference.extract_data", fake_extract)
    monkeypatch.setattr("app.flows.inference.generate_features", fake_generate)
    monkeypatch.setattr(
        "app.flows.inference.get_run_logger",
        lambda: SimpleNamespace(info=lambda _message: None),
    )

    assert cohort_extract_and_feature_task.fn(context) is features
    assert calls == ["extract", "features"]
