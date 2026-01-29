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

Settings are loaded from `config/settings.yaml` with optional local overrides in
`config/settings.local.yaml`. Switch environments with `CORTEX_ENV=dev|uat|prod`.
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
