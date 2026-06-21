from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class PostgresQueryClient:
    def __init__(self, database_url: str, connect_args: dict[str, Any] | None = None) -> None:
        if not database_url:
            raise ValueError("Postgres replica URL is not configured")
        self._engine: Engine = create_engine(database_url, connect_args=connect_args or {})

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        with self._engine.connect() as connection:
            return pd.read_sql_query(text(sql), connection, params=params or {})
