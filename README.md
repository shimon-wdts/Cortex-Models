# Cortex-Models

**Cortex‑Models** is a high‑performance machine‑learning inference service that exposes unified APIs for predictive models (for example, table‑fill and advantageous‑shoe estimators).
It is designed for **low‑latency**, **reliable**, and **real‑time decisioning** within the Cortex platform.

---

## Quick Start

### Docker (recommended)

```bash
docker build -t cortex-models .
docker run --rm -p 8000:8000 cortex-models
```


Local (Python):

macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Windows CMD:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` in your browser.

## Run the API (local)

```bash
uvicorn app.main:app --reload
```

`--reload` is for development only.

To run in a specific environment (dev/uat/prod), set `CORTEX_ENV` and use the production server settings:

```bash
export CORTEX_ENV=uat  # or dev / prod
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service listens on port 8000. Docs endpoint: `http://127.0.0.1:8000/docs`.

## API Overview

- `GET /health` — health check
- `GET /metrics` — Prometheus metrics
- `GET /alerts/fill` — list fill alerts (supports `limit`, `offset`, `start_time`, `end_time`)
- `POST /alerts/acknowledge/{alert_id}` — acknowledge an alert
- `POST /alerts/reject/{alert_id}` — reject an alert with a payload

For request/response schemas, see the interactive docs at `/docs`.

## Docker (build and run)

Build the image:

```bash
docker build -t cortex-models .
```

Run the container:

```bash
docker run --rm -p 8000:8000 cortex-models
```

Override the environment at runtime:

```bash
docker run --rm -p 8000:8000 -e CORTEX_ENV=prod cortex-models
```

## Configuration (Dynaconf)

Settings are loaded in order from `config/settings.yaml`, `config/models.yaml`,
`config/settings.{CORTEX_ENV}.yaml`, and `config/models.{CORTEX_ENV}.yaml` when
those files exist. Later files override earlier values, and nested dictionaries
are merged so a local `model_registry.postgres` override does not remove
`model_registry.models`. Switch environments with `CORTEX_ENV=dev|uat|prod`.
Environment variables with the `CORTEX_` prefix override YAML values (highest precedence).

Common environment variables:

- `CORTEX_ENV` — select the environment (`dev`, `uat`, `prod`)
- `CORTEX_*` — override any Dynaconf setting via env vars

## Kubernetes scraping (Prometheus)

If you use annotations:

```yaml
prometheus.io/scrape: "true"
prometheus.io/port: "8000"
prometheus.io/path: "/metrics"
```

If you use a ServiceMonitor (Prometheus Operator), configure a `/metrics` endpoint on port 8000.

## Inference Platform Architecture

This service separates orchestration from model business logic:

- **Prefect** is the model orchestration control plane. It owns scheduling, manual runs, retries, run state, execution history, and workers.
- **FastAPI** currently contains existing alert endpoints only; model deployment runs are triggered from Prefect UI, CLI, or API.
- **Postgres replica** is the operational data source for batch extraction.
- **Feature builders** are configurable Python callables that convert raw query results into model-ready features.
- **Model loading** is abstracted behind `app.clients.model_store`; MLflow is the default implementation.
- **Kafka** is used only to publish validated insight events after inference.

Scheduled and manually triggered runs execute the same Prefect flow code. Scheduled full-pipeline deployments default to `execution_mode=scheduled`; manual runs can override `execution_mode=manual`, and step deployments default to `manual`.

## Folder Layout

```text
app/
  api/
    routes.py             # Existing alert endpoints
  clients/
    kafka.py              # Kafka prediction publisher
    model_store.py        # MLflow/custom model loading abstraction
    postgres.py           # Postgres replica query client
  features/
    builders.py           # Feature builder callables
  flows/
    deployments.py        # Config-driven Prefect deployment helper
    inference.py          # Full pipeline and step-level Prefect flows
  inference/
    prediction_adapters.py # Inference/output event adapters
  models/
    pipeline_contracts.py # Config, pipeline, and Kafka event schemas
  pipelines/
    base.py               # Pipeline extension base class
    predicted_fills/      # Predicted fills pipeline-specific behavior
  services/
    model_registry.py     # Reads model configs from Dynaconf
    observability.py      # Structured step logs and metrics
    pipeline.py           # Testable pipeline business logic
config/
  models.yaml             # Model registry and sample PlayerPerformance config
```

## Model Config

Model behavior is driven by `config/models.yaml`, with optional environment overrides in `config/models.dev.yaml`, `config/models.uat.yaml`, and `config/models.prod.yaml`.

Each model defines:

- `enabled`
- `description` and `version`
- `type` and `source`
- default runtime `parameters`
- `schedule`
- SQL `queries`
- query `df_columns` for preserving schemas on empty query results
- feature builder path and `feature_version`
- model store provider and `model_version`
- inference adapter
- output adapter and entity mappings
- Kafka topic and key

The default sample config defines `predicted_fills` for a `PredictedFills` model with MLflow version `10.2.0`, feature version `1.0`, a cron schedule, and Kafka topic `cortex.insights.predicted-fills`.

`predicted_fills` derives query window parameters when they are not supplied:

- `end_ts` defaults to `current_time`, or can be supplied as a datetime such as `2026-06-21_10-00-00`
- `start_ts` defaults to `end_ts - start_offset_min`
- `start_offset_min` defaults to `15`
- `history_window_min` defaults to `60`
- `horizon60_min` defaults to `60`
- `lower_bound = start_ts - history_window_min`
- `upper_bound = end_ts + horizon60_min`

## Kafka Insight Event

Every published prediction is validated as an `InsightEvent` before publishing. The event carries:

- `insights_id`
- `occurred_at`
- `gaming_day`
- `source`
- `env`
- `version`
- `model.type`
- `model.version`
- `model.feature_version`
- `model.run_id`
- `entity`
- `severity`
- `payload`

The generic output adapter maps configured entity fields into the `entity` list and places model-specific output into `payload.result`, `payload.presentation`, and `payload.actions`.

## Prefect Flow Design

Full pipeline flow:

```text
cortex-model-pipeline
  extract_data_task
  feature_engineering_task
  inference_task
  publish_predictions_task
```

Step-level flow:

```text
cortex-model-step
  single_step_task(step=extract_data | feature_engineering | inference | publish)
```

Create/update deployments from config:

```bash
python -m app.flows.deployments
```

Run a local process worker:

```bash
prefect worker start -p cortex-models --type process
```

For production, use a work pool that matches the runtime platform, usually Kubernetes or Docker. Schedules should be changed through Prefect deployment configuration, not by adding scheduler code to this service.

## Manual Runs

Trigger deployment runs from Prefect UI or CLI. For example, trigger the full pipeline:

```bash
prefect deployment run 'cortex-model-pipeline/predicted_fills' \
  --param model_name=predicted_fills \
  --param execution_mode=manual \
  --param parameters='{"end_ts":"2026-06-21T10:00:00Z"}'
```

Trigger feature engineering only:

```bash
prefect deployment run 'cortex-model-step/predicted_fills-feature_engineering' \
  --param model_name=predicted_fills \
  --param step=feature_engineering \
  --param inputs='{"raw_data":{"load_tray_scans":[{"table_id":"BA0054","tray_balance":10000,"tray_ts":"2026-06-21T09:55:00Z"}]}}'
```

## Observability

Each pipeline step emits structured JSON log events:

- `model_step_started`
- `model_step_completed`
- `model_step_failed`
- `model_step_rows`

Prometheus metrics include model/run labels:

- `model_step_runs_total`
- `model_step_duration_seconds`
- `model_step_rows_total`
- `model_predictions_published_total`

Task failures are captured at the step boundary with `failure_type` and `failure_reason`, while Prefect remains the source of truth for task and flow states.

## Batch And Streaming-Like Models

Batch models should use scheduled `cortex-model-pipeline` deployments with cron, interval, or rrule schedules.

Near-real-time models should still use Prefect deployments, but usually with:

- disabled or no schedule
- manual or event-triggered runs
- tighter concurrency limits
- smaller query windows
- idempotency keys from the caller
- a dedicated Kafka topic per model group where needed

For very high-frequency streaming workloads, keep Kafka as the event distribution layer and consider a separate consumer that triggers bounded Prefect runs or performs micro-batch aggregation. Avoid adding a polling scheduler inside this project.
