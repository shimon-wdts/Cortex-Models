#!/usr/bin/env bash
set -euo pipefail

PREFECT_API_URL="${PREFECT_API_URL:-http://prefect-server:4200/api}"
PREFECT_WORK_POOL="${PREFECT_WORK_POOL:-cortex-models}"
PREFECT_WORK_POOL_TYPE="${PREFECT_WORK_POOL_TYPE:-process}"
CORTEX_ENV="${CORTEX_ENV:-local}"

export PREFECT_API_URL
export CORTEX_ENV

python - <<'PY'
import os
import sys
import time
import urllib.request

url = os.environ["PREFECT_API_URL"].rstrip("/") + "/health"
for _ in range(60):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status < 500:
                print(f"Prefect server is ready: {url}")
                sys.exit(0)
    except Exception:
        time.sleep(1)

print(f"Timed out waiting for Prefect server: {url}", file=sys.stderr)
sys.exit(1)
PY

prefect work-pool create "${PREFECT_WORK_POOL}" \
    --type "${PREFECT_WORK_POOL_TYPE}" \
    --overwrite \
    --no-prompt

python -m app.flows.deployments
