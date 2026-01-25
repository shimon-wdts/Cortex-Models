from typing import Optional

from fastapi import APIRouter
from dishka.integrations.fastapi import DishkaRoute, FromDishka

from app.clients.pas import PasClient
from app.core import config
from app.models.schemas import RejectPayload
from app.services import alerts


router = APIRouter(route_class=DishkaRoute)


@router.get("/alerts/fill")
def list_fill(
    limit: int = 12,
    offset: int = 0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    return alerts.list_fill(limit=limit, offset=offset, start_time=start_time, end_time=end_time)


@router.post("/alerts/acknowledge/{alert_id}")
def acknowledge(alert_id: str, pas: FromDishka[PasClient]):
    alerts.save_ack(alert_id, status="acknowledged", reason="", user="")

    decision_payload = alerts.build_decision_payload(alert_id, "ACCEPTED", "", "")

    try:
        pas_result = pas.publish_topic_message(config.PAS_DECISION_TOPIC, decision_payload)
    except Exception as exc:
        pas_result = {"error": str(exc)}

    return {"ok": True, "alert_id": alert_id, "status": "acknowledged", "pas": pas_result}


@router.post("/alerts/reject/{alert_id}")
def reject(alert_id: str, payload: RejectPayload, pas: FromDishka[PasClient]):
    alerts.save_ack(
        alert_id=alert_id,
        status="rejected",
        reason=payload.reason or "",
        user=payload.user or "",
    )

    decision_payload = alerts.build_decision_payload(
        alert_id,
        "REJECTED",
        payload.reason or "",
        payload.user or "",
    )

    try:
        pas_result = pas.publish_topic_message(config.PAS_DECISION_TOPIC, decision_payload)
    except Exception as exc:
        pas_result = {"error": str(exc)}

    return {
        "ok": True,
        "alert_id": alert_id,
        "status": "rejected",
        "reason": payload.reason,
        "user": payload.user,
        "timestamp": alerts.now_iso(),
        "pas": pas_result,
    }
