from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CATBOOST_DEPS, PERIODS


def enable_local_catboost() -> None:
    if CATBOOST_DEPS.exists():
        path = str(CATBOOST_DEPS)
        if path not in sys.path:
            sys.path.insert(0, path)


def normalize_path(path: str | Path) -> Path:
    text = str(path)
    if text.startswith("/mnt/c/"):
        text = "C:/" + text[len("/mnt/c/") :]
    return Path(text)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)


def write_json(obj: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def robust_pct_rank(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if valid.sum() == 0:
        return out.fillna(50.0)
    out.loc[valid] = s[valid].rank(method="average", pct=True) * 100.0
    return out.fillna(50.0)


def assign_period(gaming_day: pd.Series) -> pd.Series:
    days = pd.to_datetime(gaming_day, errors="coerce").dt.day
    out = pd.Series("", index=gaming_day.index, dtype="object")
    for label, (start, end) in PERIODS.items():
        out.loc[days.between(start, end, inclusive="both")] = label
    return out


def add_total_period(df: pd.DataFrame, period_col: str = "period") -> pd.DataFrame:
    total = df.copy()
    total[period_col] = "Total"
    return pd.concat([df, total], ignore_index=True)


def clean_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    as_int = numeric.astype("Int64")
    return as_int.astype(str).replace("<NA>", "")


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = pd.DataFrame(index=df.index)
    scalers: dict[str, dict[str, float]] = {}
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(0.0, index=df.index)
        median = float(values.median()) if values.notna().any() else 0.0
        q25 = float(values.quantile(0.25)) if values.notna().any() else median - 1.0
        q75 = float(values.quantile(0.75)) if values.notna().any() else median + 1.0
        scale = max(q75 - q25, 1.0)
        out[col] = ((values.fillna(median) - median) / scale).clip(-6, 6)
        scalers[col] = {"median": median, "scale": scale}
    return out, scalers


def apply_zscore(df: pd.DataFrame, cols: list[str], scalers: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(0.0, index=df.index)
        median = scalers.get(col, {}).get("median", 0.0)
        scale = max(float(scalers.get(col, {}).get("scale", 1.0)), 1.0)
        out[col] = ((values.fillna(median) - median) / scale).clip(-6, 6)
    return out
