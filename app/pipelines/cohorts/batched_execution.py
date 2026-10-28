from __future__ import annotations

import gc
import json
import os
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from app.clients.postgres import PostgresQueryClient
from app.models.pipeline_contracts import QueryConfig, RunContext
from app.pipelines.cohorts.source import CohortsFeatureResult, build_features_from_frames, combine_feature_batches
from app.services.model_registry import ModelRegistry


LogCallback = Callable[[str], None]


def build_cohort_features_in_batches(
    context: RunContext,
    *,
    registry: ModelRegistry,
    client: PostgresQueryClient,
    log: LogCallback,
) -> CohortsFeatureResult:
    """Extract and aggregate one player batch at a time to bound worker memory."""
    started = perf_counter()
    config = registry.get_model(context.model_name)
    queries = {query.name: query for query in config.queries}
    eligible_query = queries.get("eligible_players")
    if eligible_query is None:
        raise ValueError(f"{context.model_name} requires an eligible_players query for batched execution")

    batch_size = max(1, int(context.query_parameters.get("extract_batch_size", 100)))
    _log(
        log,
        "cohort_run_started",
        model_name=context.model_name,
        gaming_day_start=context.query_parameters.get("gaming_day_start"),
        gaming_day_end_exclusive=context.query_parameters.get("gaming_day_end_exclusive"),
        batch_size=batch_size,
        rss_mb=_process_rss_mb(),
    )

    eligible = _run_query(
        client,
        eligible_query,
        _render_params(eligible_query.params, context.query_parameters),
        log=log,
        batch_number=0,
        batch_count=0,
    )
    player_ids = eligible["PlayerId"].dropna().drop_duplicates().tolist()
    winner_loser_ratio = _winner_loser_ratio(eligible)
    batch_count = (len(player_ids) + batch_size - 1) // batch_size
    _log(
        log,
        "cohort_eligibility_completed",
        eligible_players=len(player_ids),
        winner_loser_ratio=round(winner_loser_ratio, 6),
        batch_count=batch_count,
        eligible_frame_mb=_frame_memory_mb(eligible),
        rss_mb=_process_rss_mb(),
    )

    if not player_ids:
        _log(log, "cohort_run_completed", duration_seconds=perf_counter() - started, eligible_players=0)
        return combine_feature_batches([], winner_loser_ratio=winner_loser_ratio, progress=log)

    del eligible
    gc.collect()
    dependent_queries = [query for query in config.queries if query.batch_param]
    feature_batches: list[CohortsFeatureResult] = []
    raw_rows_by_query = {query.name: 0 for query in dependent_queries}

    for batch_index, offset in enumerate(range(0, len(player_ids), batch_size), start=1):
        batch_started = perf_counter()
        batch_ids = player_ids[offset : offset + batch_size]
        _log(
            log,
            "cohort_batch_started",
            batch_number=batch_index,
            batch_count=batch_count,
            players=len(batch_ids),
            rss_mb=_process_rss_mb(),
        )

        raw_data: dict[str, pd.DataFrame] = {}
        for query in dependent_queries:
            params = _render_params(query.params, context.query_parameters)
            params[query.batch_param or "player_ids"] = batch_ids
            frame = _run_query(
                client,
                query,
                params,
                log=log,
                batch_number=batch_index,
                batch_count=batch_count,
            )
            if query.deduplicate_on and not frame.empty:
                before = len(frame)
                dedup_started = perf_counter()
                frame = frame.drop_duplicates(query.deduplicate_on, keep="last").reset_index(drop=True)
                _log(
                    log,
                    "cohort_query_deduplicated",
                    query=query.name,
                    batch_number=batch_index,
                    rows_before=before,
                    rows_after=len(frame),
                    duration_seconds=perf_counter() - dedup_started,
                    frame_mb=_frame_memory_mb(frame),
                    rss_mb=_process_rss_mb(),
                )
            raw_data[query.name] = frame
            raw_rows_by_query[query.name] += len(frame)

        _log(
            log,
            "cohort_batch_feature_engineering_started",
            batch_number=batch_index,
            sessions=len(raw_data.get("sessions", ())),
            bets=len(raw_data.get("bets", ())),
            games=len(raw_data.get("games", ())),
            raw_frame_mb=sum(_frame_memory_mb(frame) for frame in raw_data.values()),
            rss_mb=_process_rss_mb(),
        )
        progress = _batch_progress(log, batch_index, batch_count)
        feature_batch = build_features_from_frames(
            raw_sessions=raw_data.get("sessions", pd.DataFrame()),
            raw_bets=raw_data.get("bets", pd.DataFrame()),
            raw_games=raw_data.get("games", pd.DataFrame()),
            observation_start=context.query_parameters["gaming_day_start"],
            progress=progress,
            winner_loser_ratio=winner_loser_ratio,
            score_features=False,
        )
        feature_batches.append(feature_batch)
        _log(
            log,
            "cohort_batch_completed",
            batch_number=batch_index,
            batch_count=batch_count,
            aggregate_rows=len(feature_batch.player_period_features),
            aggregate_frame_mb=_frame_memory_mb(feature_batch.player_period_features),
            duration_seconds=perf_counter() - batch_started,
            rss_mb=_process_rss_mb(),
        )

        del raw_data, frame
        collected = gc.collect()
        _log(
            log,
            "cohort_batch_memory_released",
            batch_number=batch_index,
            collected_objects=collected,
            retained_aggregate_batches=len(feature_batches),
            rss_mb=_process_rss_mb(),
        )

    _log(
        log,
        "cohort_global_scoring_started",
        feature_batches=len(feature_batches),
        aggregate_rows=sum(len(batch.player_period_features) for batch in feature_batches),
        rss_mb=_process_rss_mb(),
    )
    result = combine_feature_batches(
        feature_batches,
        winner_loser_ratio=winner_loser_ratio,
        progress=log,
    )
    result.metadata.update(
        {
            "eligible_players": len(player_ids),
            "extract_batch_size": batch_size,
            "raw_rows_by_query": raw_rows_by_query,
        }
    )
    _log(
        log,
        "cohort_run_completed",
        duration_seconds=perf_counter() - started,
        eligible_players=len(player_ids),
        feature_batches=len(feature_batches),
        player_period_rows=len(result.player_period_features),
        player_total_rows=len(result.player_total_features),
        feature_frame_mb=_frame_memory_mb(result.player_period_features),
        raw_rows_by_query=raw_rows_by_query,
        rss_mb=_process_rss_mb(),
    )
    return result


def _run_query(
    client: PostgresQueryClient,
    query: QueryConfig,
    params: dict[str, Any],
    *,
    log: LogCallback,
    batch_number: int,
    batch_count: int,
) -> pd.DataFrame:
    started = perf_counter()
    _log(
        log,
        "cohort_query_started",
        query=query.name,
        batch_number=batch_number,
        batch_count=batch_count,
        player_filter_count=len(params.get(query.batch_param or "", ())) if query.batch_param else None,
        rss_mb=_process_rss_mb(),
    )
    try:
        frame = client.query(query.sql, params)
    except Exception as exc:
        _log(
            log,
            "cohort_query_failed",
            query=query.name,
            batch_number=batch_number,
            duration_seconds=perf_counter() - started,
            failure_type=type(exc).__name__,
            failure_reason=str(exc),
            rss_mb=_process_rss_mb(),
        )
        raise
    if frame.empty:
        frame = pd.DataFrame(columns=query.df_columns)
    _log(
        log,
        "cohort_query_completed",
        query=query.name,
        batch_number=batch_number,
        batch_count=batch_count,
        rows=len(frame),
        columns=len(frame.columns),
        duration_seconds=perf_counter() - started,
        frame_mb=_frame_memory_mb(frame),
        rss_mb=_process_rss_mb(),
    )
    return frame


def _winner_loser_ratio(eligible: pd.DataFrame) -> float:
    if "CasinoWinTotal" not in eligible.columns:
        raise ValueError("eligible_players query must return CasinoWinTotal for population-consistent batching")
    player_actual = pd.to_numeric(eligible["CasinoWinTotal"], errors="coerce").fillna(0.0)
    winners = int(player_actual.lt(0).sum())
    losers = int(player_actual.gt(0).sum())
    return winners / losers if losers else 0.25


def _render_params(params: dict[str, Any], runtime_params: dict[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            rendered[key] = runtime_params[value[2:-2].strip()]
        else:
            rendered[key] = value
    return rendered


def _batch_progress(log: LogCallback, batch_number: int, batch_count: int) -> LogCallback:
    def emit(message: str) -> None:
        log(
            f"{message} batch_number={batch_number} batch_count={batch_count} "
            f"rss_mb={_process_rss_mb()}"
        )

    return emit


def _frame_memory_mb(frame: pd.DataFrame) -> float:
    return round(float(frame.memory_usage(index=True, deep=True).sum()) / (1024 * 1024), 3)


def _process_rss_mb() -> float | None:
    try:
        import psutil

        return round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024), 3)
    except (ImportError, OSError):
        pass

    if os.name == "posix":
        try:
            with open("/proc/self/statm", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            return round(resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 3)
        except (OSError, ValueError, IndexError):
            return None

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            process = get_current_process()
            if get_process_memory_info(process, ctypes.byref(counters), counters.cb):
                return round(float(counters.WorkingSetSize) / (1024 * 1024), 3)
        except (AttributeError, OSError, ValueError):
            return None
    return None


def _log(log: LogCallback, event: str, **fields: Any) -> None:
    payload = {"event": event, **{key: value for key, value in fields.items() if value is not None}}
    log(json.dumps(payload, default=str, sort_keys=True))
