from __future__ import annotations

from threading import Lock
from typing import Any

import pandas as pd

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
