from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def recommendation_deduplication(
    *,
    policy_id: str,
    entity_type: str,
    action_fields: Iterable[str],
    scope_entity_types: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the common recommendation-level deduplication contract."""
    match_keys = [
        f"entity[type={entity_type}].id",
        *(f"entity[type={scope_type}].id" for scope_type in scope_entity_types),
        "$context.target_application",
        "payload.result.decision_class",
        *(f"recommendation.action.{field}" for field in action_fields),
    ]
    return {
        "policy_id": policy_id,
        "duplication_logic": [
            {
                "key": key,
                "match_type": "same",
                "evaluation": "exact_match",
            }
            for key in match_keys
        ],
        "prior_conditions": [
            {
                "key": "recommendation.time_to_live",
                "operator": ">",
                "value": "$context.processing_time",
            },
            {
                "key": "repository.suppressed",
                "operator": "=",
                "value": False,
            },
            {
                "key": "repository.delivery_status",
                "operator": "in",
                "value": ["pending", "delivered"],
            },
        ],
        "on_match": "suppress",
        "extend_suppression_from_duplicates": False,
    }
