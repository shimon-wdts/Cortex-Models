from __future__ import annotations

from threading import Lock
from types import SimpleNamespace
from typing import Any

import pandas as pd

from app.models.pipeline_contracts import ExecutionMode, QueryConfig, RunContext
from app.pipelines.cohorts.batched_execution import build_cohort_features_in_batches
from app.pipelines.cohorts.source import CohortsFeatureResult
from app.services.pipeline import _batch_values, _query_in_batches


class RecordingQueryClient:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self._lock = Lock()

    def query(self, _sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        player_ids = list((params or {})["player_ids"])
        with self._lock:
            self.calls.append(player_ids)
        return pd.DataFrame({"PlayerId": player_ids})


def test_batch_values_are_unique_and_ignore_nulls() -> None:
    data = {"eligible_players": pd.DataFrame({"PlayerId": [1, 2, 2, None, 3]})}

    assert _batch_values(data, "eligible_players", "PlayerId") == [1.0, 2.0, 3.0]


def test_query_in_batches_fetches_every_player_without_an_output_cap() -> None:
    client = RecordingQueryClient()

    frame = _query_in_batches(
        client=client,
        sql="SELECT player_id WHERE player_id = ANY(:player_ids)",
        params={"window": 21},
        batch_param="player_ids",
        values=[1, 2, 3, 4, 5],
        batch_size=2,
        workers=2,
        columns=["PlayerId"],
    )

    assert sorted(frame["PlayerId"].tolist()) == [1, 2, 3, 4, 5]
    assert sorted(client.calls) == [[1, 2], [3, 4], [5]]


def test_query_in_batches_skips_database_when_no_players_are_eligible() -> None:
    client = RecordingQueryClient()

    frame = _query_in_batches(
        client=client,
        sql="SELECT player_id WHERE player_id = ANY(:player_ids)",
        params={},
        batch_param="player_ids",
        values=[],
        batch_size=500,
        workers=4,
        columns=["PlayerId"],
    )

    assert frame.empty
    assert frame.columns.tolist() == ["PlayerId"]
    assert client.calls == []


def test_cohort_execution_aggregates_and_releases_each_player_batch(monkeypatch) -> None:
    queries = [
        QueryConfig(
            name="eligible_players",
            sql="eligible_players",
            params={},
            df_columns=["PlayerId", "BetCount", "CasinoWinTotal"],
        ),
        *[
            QueryConfig(
                name=name,
                sql=name,
                params={},
                df_columns=["PlayerId"],
                batch_source_query="eligible_players",
                batch_source_column="PlayerId",
                batch_param="player_ids",
            )
            for name in ("bets", "sessions")
        ],
    ]
    registry = SimpleNamespace(get_model=lambda _name: SimpleNamespace(queries=queries))

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[int]]] = []

        def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
            if sql == "eligible_players":
                return pd.DataFrame(
                    {
                        "PlayerId": [1, 2, 3],
                        "BetCount": [30, 40, 50],
                        "CasinoWinTotal": [-10.0, 20.0, 0.0],
                    }
                )
            player_ids = list((params or {})["player_ids"])
            self.calls.append((sql, player_ids))
            return pd.DataFrame({"PlayerId": player_ids})

    def fake_build_features(*, raw_bets, **_kwargs):
        period = pd.DataFrame(
            {
                "PlayerId": raw_bets["PlayerId"].astype(str),
                "period": "Total",
            }
        )
        return CohortsFeatureResult(period, period.copy())

    def fake_combine(batches, *, winner_loser_ratio, progress):
        assert winner_loser_ratio == 1.0
        assert callable(progress)
        period = pd.concat([batch.player_period_features for batch in batches], ignore_index=True)
        period = period.rename(columns={"PlayerId": "player_id"})
        return CohortsFeatureResult(period, period.copy())

    logs: list[str] = []
    client = Client()
    monkeypatch.setattr(
        "app.pipelines.cohorts.batched_execution.build_features_from_frames",
        fake_build_features,
    )
    monkeypatch.setattr(
        "app.pipelines.cohorts.batched_execution.combine_feature_batches",
        fake_combine,
    )
    monkeypatch.setattr("app.pipelines.cohorts.batched_execution._process_rss_mb", lambda: 123.0)
    context = RunContext(
        model_name="cohort",
        run_id="run-1",
        execution_mode=ExecutionMode.MANUAL,
        query_parameters={
            "gaming_day_start": "2026-08-17",
            "gaming_day_end_exclusive": "2026-09-07",
            "extract_batch_size": 2,
        },
    )

    result = build_cohort_features_in_batches(
        context,
        registry=registry,
        client=client,
        log=logs.append,
    )

    assert result.player_total_features["player_id"].tolist() == ["1", "2", "3"]
    for query_name in ("bets", "sessions"):
        assert (query_name, [1, 2]) in client.calls
        assert (query_name, [3]) in client.calls
    assert any('"event": "cohort_query_completed"' in message for message in logs)
    assert any('"event": "cohort_batch_memory_released"' in message for message in logs)
    assert any('"event": "cohort_run_completed"' in message for message in logs)
