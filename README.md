# Cortex-Models

`Cortex-Models` runs model inference pipelines for the Cortex platform. The current operational path is Prefect-first: Prefect owns scheduling, manual runs, task retries, flow history, and workers; the model code owns extraction, feature engineering, inference, and publishing.

## Quick Start

This project uses `uv` for dependency management. `uv sync` reads `pyproject.toml` and `uv.lock`, creates or updates the local `.venv`, and installs the pinned dependency set.

### macOS/Linux

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment and install dependencies:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync
```

Check the install:

```bash
.venv/bin/python --version
.venv/bin/prefect version
```

### Windows PowerShell

Install `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create the virtual environment and install dependencies:

```powershell
uv venv --python 3.13
.venv\Scripts\Activate.ps1
uv sync
```

Check the install:

```powershell
.venv\Scripts\python.exe --version
.venv\Scripts\prefect.exe version
```

### Windows CMD

Install `uv`:

```bat
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create the virtual environment and install dependencies:

```bat
uv venv --python 3.13
.venv\Scripts\activate.bat
uv sync
```

Check the install:

```bat
.venv\Scripts\python.exe --version
.venv\Scripts\prefect.exe version
```

## Run Prefect Server And Worker

Use PostgreSQL for the Prefect server database. SQLite can work for very small local tests, but Prefect server services and worker polling can produce repeated `sqlite3.OperationalError: database is locked` errors.

### 1. Start A PostgreSQL DB For Prefect

Example with Docker:

```bash
docker run --name prefect-db \
  -e POSTGRES_USER=prefect \
  -e POSTGRES_PASSWORD=prefect \
  -e POSTGRES_DB=prefect \
  -p 5433:5432 \
  -d postgres:16
```

If the container already exists:

```bash
docker start prefect-db
```

### 2. Start The Prefect Server

macOS/Linux:

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
export PREFECT_API_DATABASE_CONNECTION_URL=postgresql+asyncpg://prefect:prefect@127.0.0.1:5433/prefect

.venv/bin/prefect server start
```

Windows PowerShell:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:PREFECT_API_DATABASE_CONNECTION_URL = "postgresql+asyncpg://prefect:prefect@127.0.0.1:5433/prefect"

.venv\Scripts\prefect.exe server start
```

Windows CMD:

```bat
set PREFECT_API_URL=http://127.0.0.1:4200/api
set PREFECT_API_DATABASE_CONNECTION_URL=postgresql+asyncpg://prefect:prefect@127.0.0.1:5433/prefect

.venv\Scripts\prefect.exe server start
```

Open the Prefect UI at `http://127.0.0.1:4200`.

### 3. Create Or Update Deployments

Run this in a second terminal after the Prefect server is healthy:

macOS/Linux:

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
export CORTEX_ENV=local

.venv/bin/python -m app.flows.deployments
```

Windows PowerShell:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:CORTEX_ENV = "local"

.venv\Scripts\python.exe -m app.flows.deployments
```

Windows CMD:

```bat
set PREFECT_API_URL=http://127.0.0.1:4200/api
set CORTEX_ENV=local

.venv\Scripts\python.exe -m app.flows.deployments
```

The deployment helper reads `config/models.yaml` and environment overrides, ensures the configured work pool exists, and deploys enabled models.

### 4. Start The Worker

Run the worker in another terminal:

macOS/Linux:

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
export CORTEX_ENV=local

.venv/bin/prefect worker start -p cortex-models --type process
```

Windows PowerShell:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:CORTEX_ENV = "local"

.venv\Scripts\prefect.exe worker start -p cortex-models --type process
```

Windows CMD:

```bat
set PREFECT_API_URL=http://127.0.0.1:4200/api
set CORTEX_ENV=local

.venv\Scripts\prefect.exe worker start -p cortex-models --type process
```

The worker only needs `PREFECT_API_URL`; the PostgreSQL connection string is only required by the Prefect server process.

### 5. Trigger A Flow Run

From the Prefect UI, open the `cortex-model-pipeline/predicted_fills` deployment and click **Run**.

From the CLI:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
.venv/bin/prefect deployment run 'cortex-model-pipeline/predicted_fills' \
  --param model_name=predicted_fills \
  --param execution_mode=manual \
  --param parameters='{"end_ts":"2026-05-10 12-00-00","json_limit":3}'
```

## Configuration

Settings are loaded in this order:

```text
config/settings.yaml
config/models.yaml
config/settings.{CORTEX_ENV}.yaml
config/models.{CORTEX_ENV}.yaml
```

Later files override earlier values. Nested dictionaries are merged, so `config/settings.local.yaml` can override `model_registry.postgres.replica_url` without replacing the entire model registry.

Common environment variables:

- `CORTEX_ENV` selects local, dev, uat, or prod overrides.
- `PREFECT_API_URL` points clients and workers at the Prefect server API.
- `PREFECT_API_DATABASE_CONNECTION_URL` configures the Prefect server database.
- `CORTEX_*` values can override Dynaconf settings.

## Prefect With PostgreSQL

Recommended local Prefect server DB:

```bash
PREFECT_API_DATABASE_CONNECTION_URL=postgresql+asyncpg://prefect:prefect@127.0.0.1:5433/prefect
```

Use PostgreSQL for any shared, long-running, or frequently polled Prefect setup. SQLite is sensitive to concurrent writes from the Prefect server, worker, UI, and background services.

Important boundaries:

- The Prefect server uses `PREFECT_API_DATABASE_CONNECTION_URL`.
- The Prefect worker uses `PREFECT_API_URL`.
- The model pipeline source database is configured separately under `model_registry.postgres.replica_url`.

## Inference Platform Architecture

The inference platform separates orchestration, model runtime, and publishing:

- **Prefect server** stores deployments, schedules, flow runs, task runs, logs, and state transitions.
- **Prefect worker** polls the `cortex-models` work pool and executes flow runs in local processes.
- **Deployment helper** in `app.flows.deployments` reads model config and creates Prefect deployments for enabled models.
- **Pipeline flow** in `app.flows.inference` executes extraction, feature engineering, inference, and publishing as Prefect tasks.
- **Model registry** in `app.services.model_registry` loads model definitions from Dynaconf.
- **Operational Postgres replica** is the source for model input queries.
- **Model store** in `app.clients.model_store` loads MLflow or custom model artifacts.
- **Kafka publisher** emits final prediction or insight payloads to configured topics.
- **Observability** is split between Prefect state/logs and application-level structured metrics.

## Folder Layout

```text
app/
  clients/
    kafka.py               # Kafka prediction publisher
    model_store.py         # MLflow/custom model loading abstraction
    postgres.py            # Postgres query client
  flows/
    deployments.py         # Config-driven Prefect deployment helper
    inference.py           # Full pipeline Prefect flow and tasks
  inference/
    prediction_adapters.py # Insight-event adapter helpers
  models/
    pipeline_contracts.py  # Config, run context, and event schemas
  pipelines/
    base.py                # Pipeline extension base class
    predicted_fills/       # Predicted fills model-specific logic
  services/
    model_registry.py      # Reads model configs from Dynaconf
    observability.py       # Structured step logs and metrics
    pipeline.py            # Testable pipeline business logic
config/
  models.yaml              # Model registry and deployment config
  settings.local.yaml      # Local environment overrides
```

## Model Config

Model behavior is driven by `config/models.yaml`, with optional environment overrides in `config/models.dev.yaml`, `config/models.uat.yaml`, and `config/models.prod.yaml`.

The top-level `model_registry` contains:

- `prefect`: work pool name, work pool type, work queue, and deployment settings.
- `postgres`: source database connection for model extraction.
- `kafka`: Kafka producer bootstrap servers and producer options.
- `models`: one config block per model.

Each model config defines:

- `enabled`, `description`, `version`, `features_version`, `type`, and `source`.
- `parameters` used as default runtime parameters.
- `schedule` with `enabled`, `type`, `cron` or interval/rrule fields, timezone, and concurrency limit.
- `queries`, including SQL, templated params, and `df_columns` for empty result preservation.
- `model_store` provider, tracking URI, registered model name, version, URI, or custom loader.
- `output` adapter, entity mappings, actions, and output schema behavior.
- `kafka.topic` and `kafka.key_field`.

The current `predicted_fills` model:

- Uses the `PredictedFills` pipeline implementation.
- Reads tray scans, chip inventory, chip updates, bets, and topology data.
- Loads MLflow model version `10.2.0` by default.
- Publishes to Kafka topic `cortex-floor-insights`.
- Uses `key_field: table_id`, so Kafka records are keyed by table when that field is present.

Runtime parameters can be overridden when a flow run is triggered. `predicted_fills` accepts parameters such as:

- `end_ts`
- `start_offset_min`
- `snapshot_minute`
- `threshold`
- `json_limit`
- `opportunistic_min_prob`
- `avoided_future_trip_minutes`
- `same_pit_extra_stop_minutes`
- `max_extra_stops_per_route`

## Kafka Publishing

The publish step sends prediction records to the configured Kafka topic. The Kafka key is derived from `kafka.key_field`.

If the configured key field matches an entity type, the publisher uses that entity id. Otherwise it falls back to a top-level field of the same name. For example, with `key_field: table_id`, records with a top-level `table_id` are keyed by that value.

## Prefect Flow Design

`full_pipeline_flow` is the main deployment flow:

```text
cortex-model-pipeline (cohort, playerscore, cohort_tier_lift)
  extract-and-feature-engineering
  inference
  publish-predictions

cortex-model-pipeline (other models)
  extract-data
  feature-engineering
  inference
  publish-predictions
```

Flow behavior:

- `create_run_context` resolves model name, execution mode, runtime parameters, query parameters, and model-store path.
- `extract-and-feature-engineering` keeps the large raw cohort frames within one task and returns only aggregated player features.
- Other models retain separate `extract-data` and `feature-engineering` tasks.
- `inference` loads or runs the model-specific inference path.
- `publish-predictions` publishes final records to Kafka and records publish metrics.

Deployments are created by:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
CORTEX_ENV=local \
.venv/bin/python -m app.flows.deployments
```

The deployment helper currently deploys enabled full-pipeline model deployments. Step-level deployments are not active in the current code path.

For production, use a work pool that matches the runtime platform, usually Kubernetes or Docker. Schedules should be changed through Prefect deployment configuration, not by adding a scheduler loop inside this service.

## Manual Runs

Trigger deployment runs from the Prefect UI or CLI.

Minimal full-pipeline run:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
.venv/bin/prefect deployment run 'cortex-model-pipeline/predicted_fills' \
  --param model_name=predicted_fills \
  --param execution_mode=manual
```

Run with model parameters:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
.venv/bin/prefect deployment run 'cortex-model-pipeline/predicted_fills' \
  --param model_name=predicted_fills \
  --param execution_mode=manual \
  --param parameters='{"end_ts":"2026-05-10 12-00-00","json_limit":3}'
```

Inspect recent task runs:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
.venv/bin/prefect task-run ls --limit 20
```

Watch a flow run:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api \
.venv/bin/prefect flow-run watch <flow-run-id>
```

## Observability

Prefect is the source of truth for orchestration state:

- deployment state
- scheduled, pending, running, completed, failed, or cancelled flow runs
- task retries and task states
- worker assignment and infrastructure pid
- run logs

Application code emits structured JSON step events:

- `model_step_started`
- `model_step_completed`
- `model_step_failed`
- `model_step_rows`

Prometheus metrics include model and run labels:

- `model_step_runs_total`
- `model_step_duration_seconds`
- `model_step_rows_total`
- `model_predictions_published_total`

Task failures include `failure_type` and `failure_reason` in structured logs. Use Prefect task-run state for orchestration diagnosis and application structured events for model-step diagnosis.

## Batch And Streaming-Like Models

Batch models should use scheduled `cortex-model-pipeline` deployments with cron, interval, or rrule schedules. Use `schedule.enabled: true` only when the model should run automatically.

Near-real-time models can still run through Prefect, but should usually use:

- no schedule, or a short controlled interval
- explicit API/UI/CLI/event-triggered runs
- tight concurrency limits
- small query windows
- idempotency keys from the caller where runs may be retried
- model-specific Kafka topics when consumers need isolation

For high-frequency streaming workloads, keep Kafka as the distribution layer and use a separate consumer or micro-batch service to decide when to trigger bounded Prefect runs. Avoid adding a polling loop inside this project.
