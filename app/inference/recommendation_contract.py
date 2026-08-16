from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def recommendation_deduplication_id(identity_fields: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 identifier for a recommendation's business identity."""
    if not identity_fields:
        raise ValueError("identity_fields must not be empty")
    if any(value is None for value in identity_fields.values()):
        raise ValueError("identity_fields must not contain null values")

    canonical = json.dumps(
        dict(identity_fields),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
