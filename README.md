# Cortex-Models
Cortex‑Models is a high‑performance ML inference service providing fast, unified APIs for predictive models such as table‑fill and advantageous‑shoe estimators. Built for reliability and real‑time decisioning, it powers Cortex‑platform applications with low‑latency, scalable predictions.

## Run the API

```bash
uvicorn app.main:app --reload
```

## Configuration (Dynaconf)

Settings are loaded from `config/settings.yaml` with optional local overrides in
`config/settings.local.yaml`. Switch environments with `CORTEX_ENV=dev|uat|prod`.
Environment variables with the `CORTEX_` prefix override YAML values (highest precedence).
