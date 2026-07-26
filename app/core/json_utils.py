from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from pydantic import BaseModel


def sanitize_json(value: Any) -> Any:
    """Return a recursively JSON-safe value, replacing non-finite numbers with None."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [sanitize_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return sanitize_json(value.item())

    if isinstance(value, BaseModel):
        return sanitize_json(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_json(asdict(value))

    if isinstance(value, Mapping):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_json(item) for item in value)

    if isinstance(value, Enum):
        return sanitize_json(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)

    return value
