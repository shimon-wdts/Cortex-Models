import asyncio
import time

from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

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

_START_TIME = time.time()

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
    # Standard HTTP metrics via instrumentator (requests, latency, in-flight).
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
