import asyncio
import logging
import time

from prometheus_client import Counter, Gauge, Histogram

APP_EXCEPTIONS_TOTAL = Counter(
    "app_exceptions_total",
    "Total handled exceptions.",
    labelnames=("exception_type", "route"),
)
EVENT_LOOP_LAG_SECONDS = Gauge(
    "python_event_loop_lag_seconds",
    "Event loop lag in seconds.",
)
APP_UPTIME_SECONDS = Gauge(
    "app_uptime_seconds",
    "App uptime in seconds.",
)
MODEL_STEP_RUNS_TOTAL = Counter(
    "model_step_runs_total",
    "Model pipeline step executions by model, run, step, and status.",
    labelnames=("model_name", "run_id", "step", "status"),
)
MODEL_STEP_DURATION_SECONDS = Histogram(
    "model_step_duration_seconds",
    "Model pipeline step duration by model, run, step, and status.",
    labelnames=("model_name", "run_id", "step", "status"),
)
MODEL_STEP_ROWS_TOTAL = Counter(
    "model_step_rows_total",
    "Rows processed by model pipeline step.",
    labelnames=("model_name", "run_id", "step"),
)
MODEL_PREDICTIONS_PUBLISHED_TOTAL = Counter(
    "model_predictions_published_total",
    "Prediction events published to Kafka by model, run, and topic.",
    labelnames=("model_name", "run_id", "topic"),
)

_START_TIME = time.time()
logger = logging.getLogger(__name__)

async def monitor_event_loop_lag(stop_event: asyncio.Event, interval: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    next_time = loop.time() + interval
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        now = loop.time()
        lag = max(0.0, now - next_time)
        EVENT_LOOP_LAG_SECONDS.set(lag)
        APP_UPTIME_SECONDS.set(time.time() - _START_TIME)
        next_time = now + interval

def route_label(route) -> str:
    if route is None:
        return "unknown"
    path = getattr(route, "path", None)
    return path or "unknown"

def setup_metrics(app) -> None:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ModuleNotFoundError:
        logger.warning("prometheus_fastapi_instrumentator is not installed; /metrics HTTP instrumentation is disabled")
        return

    # Standard HTTP metrics via instrumentator (requests, latency, in-flight).
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
