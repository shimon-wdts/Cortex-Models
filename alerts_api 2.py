# api/alerts_api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
import pandas as pd
import os
import json
from datetime import datetime, timezone
import time
import requests

# ================================
# UTILITIES (must come first)
# ================================
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_decision_payload(alert_id: str, status: str, reason: str, user: str):
    return {
        "decisionId": f"dec-{alert_id}-{int(time.time())}",
        "alertId": alert_id,
        "decision": status.upper(),  # ACCEPTED / REJECTED
        "reason": reason or "",
        "user": user or "",
        "decidedAt": _now_iso(),
    }


# ================================
# PAS CLIENT
# ================================
class PasClient:
    def __init__(self, base_url: str, partner_id: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.partner_id = partner_id
        self.api_key = api_key
        self._access_token = None
        self._refresh_token = None
        self._access_expiry = 0

    # -----------------------
    # AUTH
    # -----------------------
    def _issue_token(self):
        url = f"{self.base_url}/api/pas/v1/auth/{self.partner_id}/token/issue"
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}

        resp = requests.post(url, headers=headers, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        self._access_token = data["accessToken"]
        self._refresh_token = data["refreshToken"]
        self._access_expiry = time.time() + 240  # refresh 1 min before expiry

    def _refresh_access_token(self):
        url = f"{self.base_url}/api/pas/v1/auth/{self.partner_id}/token/refresh"
        headers = {"Content-Type": "application/json"}

        resp = requests.post(
            url,
            json={"refreshToken": self._refresh_token},
            headers=headers,
            timeout=10,
        )

        if resp.status_code != 200:
            self._issue_token()
            return

        data = resp.json()
        self._access_token = data["accessToken"]
        self._access_expiry = time.time() + 240

    def _ensure_token(self):
        if not self._access_token or time.time() >= self._access_expiry:
            if not self._refresh_token:
                self._issue_token()
            else:
                self._refresh_access_token()

    def _headers(self):
        self._ensure_token()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    # -----------------------
    # PUBLISH TO TOPIC
    # -----------------------
    def publish_topic_message(self, topic_name: str, payload: dict):
        url = f"{self.base_url}/api/pas/v1/pubsub/{self.partner_id}/topics/{topic_name}/messages"
        resp = requests.post(url, headers=self._headers(), json={"payload": payload}, timeout=10)
        resp.raise_for_status()
        return resp.json()


# ================================
# LOCAL CONFIG
# ================================
DATA_DIR = r"C:\Users\Yoni Adler\Documents\insightsAI\data"

ALERTS_CSV_V10 = os.path.join(DATA_DIR, "predictive_fill_top_alerts_v10.csv")
ACK_FILE = os.path.join(DATA_DIR, "acknowledged_alerts_v10.csv")

PAS_BASE_URL = "https://YOUR-PAS-ENDPOINT"
PAS_PARTNER_ID = "your-partner-id"
PAS_API_KEY = "your-api-key"
PAS_DECISION_TOPIC = "inspection-decisions"

pas = PasClient(
    base_url=PAS_BASE_URL,
    partner_id=PAS_PARTNER_ID,
    api_key=PAS_API_KEY,
)

# ================================
# FASTAPI APP
# ================================
app = FastAPI(
    title="Casino Intervention API",
    version="v10",
    description="Predicted chip-fill decisions, ROI, and rationale (v10).",
)

# ================================
# MODELS
# ================================
class Rationale(BaseModel):
    rel_balance: Optional[float] = None
    exp_payout_next30: Optional[float] = None
    exp_payout_hr: Optional[float] = None
    safety_buffer: Optional[float] = None
    model: Optional[str] = None
    pretty: Optional[str] = None
    _raw: Optional[str] = None


class FillCard(BaseModel):
    alert_id: str
    table_id: str
    entity_label: str
    alert_type: str
    payout_risk_label: str
    updated: str

    severity: str
    option_count: int
    types: List[str]
    modeled_impact_per_hr: List[float]
    time_to_depletion_min: List[int]
    roi: List[Optional[float]]
    roi_unit: List[Optional[str]]
    trigger_metric: str
    recommendation: List[str]

    chart_series: List[List[List[float]]]
    chart_labels: List[str]
    x_labels: List[str]
    chart_label: str = "EV/hr"

    rationale: List[Rationale]

    # ✅ NEW FIELDS
    isExportable: bool = False
    exportable_content: Optional[Any] = None
    isActionable: bool = False


# ================================
# HELPERS
# ================================
ACK_COLUMNS = ["alert_id", "ack_time", "status", "reason", "user"]


def _load_alerts_df() -> pd.DataFrame:
    if not os.path.exists(ALERTS_CSV_V10):
        return pd.DataFrame()
    df = pd.read_csv(ALERTS_CSV_V10)
    dupes = [c for c in df.columns if c.endswith(".1") or c.endswith(".2")]
    df = df.drop(columns=dupes, errors="ignore")
    return df


def _load_ack_df() -> pd.DataFrame:
    if os.path.exists(ACK_FILE):
        df = pd.read_csv(ACK_FILE)
        for col in ACK_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[ACK_COLUMNS]
    return pd.DataFrame(columns=ACK_COLUMNS)


def _save_ack(alert_id: str, status: str, reason: str = "", user: str = "") -> None:
    df = _load_ack_df()
    df.loc[len(df)] = [alert_id, _now_iso(), status, reason, user]
    df.to_csv(ACK_FILE, index=False)


def _payout_risk_label(rel, buf) -> str:
    try:
        rel = float(rel)
        buf = float(buf)
    except Exception:
        return "Status unknown"

    if rel <= -3 * (buf or 1):
        return "Severely under buffer"
    if rel <= -1 * (buf or 1):
        return "Under buffer"
    if rel < 0:
        return "Near buffer"
    return "Above buffer"


def _safe_json_load(s: Any, default):
    if isinstance(s, str) and s.strip():
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


def _severity_from_ttd(ttd: float) -> str:
    try:
        ttd = float(ttd)
    except Exception:
        return "Unknown"
    if ttd <= 45:
        return "Critical"
    if ttd <= 90:
        return "High"
    if ttd <= 180:
        return "Medium"
    return "Low"


def _pretty_rationale(r: dict) -> str:
    parts = []
    try:
        rb = float(r.get("rel_balance", 0))
        if rb < 0:
            parts.append(f"Balance is {abs(rb):,.0f} below buffer")
        elif rb > 0:
            parts.append(f"Balance is {rb:,.0f} above buffer")
    except Exception:
        pass

    try:
        parts.append(f"Expected payout next 30 min: {float(r.get('exp_payout_next30', 0)):,.0f}")
    except Exception:
        pass

    try:
        parts.append(f"EV/hr impact: {float(r.get('exp_payout_hr', 0)):,.0f}")
    except Exception:
        pass

    try:
        parts.append(f"Safety buffer: {float(r.get('safety_buffer', 0)):,.0f}")
    except Exception:
        pass

    return " | ".join(parts) if parts else "Model rationale unavailable"


# -----------------------
# ✅ NEW: Export helpers
# -----------------------
def _is_exportable_alert(row: pd.Series) -> bool:
    """
    Decide whether this alert should be exportable.
    Minimal rule: if it has core fields required for a CSV row.
    Adjust as needed (or drive this by a CSV column).
    """
    required = ["alert_id", "table_id", "updated", "predicted_time_to_depletion_min"]
    return all(str(row.get(k, "")).strip() != "" for k in required)


def _build_exportable_content(row: pd.Series, card: FillCard) -> dict:
    """
    A compact, CSV-friendly payload. Web can flatten/stringify as needed.
    """
    rec_type = card.types[0] if card.types else None
    rec_ttd = card.time_to_depletion_min[0] if card.time_to_depletion_min else None
    rec_impact = card.modeled_impact_per_hr[0] if card.modeled_impact_per_hr else None
    rec_roi = card.roi[0] if card.roi else None
    rec_roi_unit = card.roi_unit[0] if card.roi_unit else None
    rec_text = card.recommendation[0] if card.recommendation else None
    rec_rationale = card.rationale[0].pretty if card.rationale else None

    return {
        "alert_id": card.alert_id,
        "table_id": card.table_id,
        "entity_label": card.entity_label,
        "alert_type": card.alert_type,
        "severity": card.severity,
        "updated": card.updated,
        "payout_risk_label": card.payout_risk_label,
        "trigger_metric": card.trigger_metric,
        "recommended_type": rec_type,
        "recommended_time_to_depletion_min": rec_ttd,
        "recommended_modeled_impact_per_hr": rec_impact,
        "recommended_roi": rec_roi,
        "recommended_roi_unit": rec_roi_unit,
        "recommended_recommendation": rec_text,
        "recommended_rationale": rec_rationale,
        # Optional extra raw fields from the row
        "predicted_time_to_depletion_min": row.get("predicted_time_to_depletion_min"),
        "expected_deficit": row.get("expected_deficit"),
    }


# -----------------------
# Card builder
# -----------------------
def _row_to_card(row: pd.Series) -> FillCard:
    types = _safe_json_load(row.get("types_json"), [])
    impacts = _safe_json_load(row.get("modeled_impact_per_hr_json"), [])
    ttd_list = _safe_json_load(row.get("time_to_depletion_min_json"), [])
    roi_list = _safe_json_load(row.get("roi_json"), [])
    recs = _safe_json_load(row.get("recommendation_json"), [])
    chart_series_raw = _safe_json_load(row.get("chart_series_json"), [])
    rationale_raw = _safe_json_load(row.get("rationale_json"), [])

    roi_unit_list = _safe_json_load(row.get("roi_unit_json"), [])
    if not roi_unit_list and roi_list:
        roi_unit_list = ["x" if abs(float(v or 0)) <= 100 else "m" for v in roi_list]

    rationale_objs: List[Rationale] = []
    for r in rationale_raw:
        if isinstance(r, dict):
            rationale_objs.append(
                Rationale(
                    rel_balance=r.get("rel_balance"),
                    exp_payout_next30=r.get("exp_payout_next30"),
                    exp_payout_hr=r.get("exp_payout_hr"),
                    safety_buffer=r.get("safety_buffer"),
                    model=r.get("model"),
                    pretty=_pretty_rationale(r),
                    _raw=str(r),
                )
            )
        else:
            rationale_objs.append(Rationale(pretty=str(r), _raw=str(r)))

    rel = rationale_objs[0].rel_balance if rationale_objs else row.get("rel_balance")
    buf = rationale_objs[0].safety_buffer if rationale_objs else row.get("safety_buffer")
    payout_label = _payout_risk_label(rel, buf)

    processed_chart_series: List[List[List[float]]] = []
    for series in chart_series_raw:
        if isinstance(series, list) and series and isinstance(series[0], list):
            processed_chart_series.append([[float(x) for x in s] for s in series])
        elif isinstance(series, list):
            one = [float(x) for x in series]
            processed_chart_series.append([one, one])
        else:
            processed_chart_series.append([[], []])

    x_labels_raw = _safe_json_load(row.get("chart_x_labels_json"), [])
    if x_labels_raw:
        x_labels = [str(x) for x in x_labels_raw]
    else:
        n = len(processed_chart_series[0][0]) if processed_chart_series else 0
        x_labels = [f"Point {i+1}" for i in range(n)]

    severity = _severity_from_ttd(row.get("predicted_time_to_depletion_min"))
    option_count = int(row.get("option_count") or len(types))

    card = FillCard(
        alert_id=str(row.get("alert_id")),
        table_id=str(row.get("table_id")),
        entity_label=f"TABLE {row.get('table_id')}",
        alert_type=str(row.get("alert_type") or "ChipDepletion"),
        payout_risk_label=payout_label,
        updated=str(row.get("updated") or _now_iso()),
        severity=severity,
        option_count=option_count,
        types=types[:option_count],
        modeled_impact_per_hr=impacts[:option_count],
        time_to_depletion_min=ttd_list[:option_count],
        roi=roi_list[:option_count],
        roi_unit=roi_unit_list[:option_count],
        trigger_metric=str(row.get("trigger_metric") or ""),
        recommendation=recs[:option_count],
        chart_series=processed_chart_series[:option_count],
        chart_labels=["EV/hr (recommended)", "EV/hr (baseline)"],
        x_labels=x_labels,
        rationale=rationale_objs[:option_count],
    )

    # ✅ NEW: isActionable
    # This service supports both /alerts/acknowledge/{id} and /alerts/reject/{id},
    # so we can mark these alerts actionable.
    card.isActionable = True

    # ✅ NEW: isExportable + exportable_content
    card.isExportable = _is_exportable_alert(row)
    card.exportable_content = _build_exportable_content(row, card) if card.isExportable else None

    return card


# ================================
# ENDPOINT: LIST ALERTS
# ================================
@app.get("/alerts/fill")
def list_fill(
    limit: int = 12,
    offset: int = 0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    df = _load_alerts_df()
    if df.empty:
        return {"total": 0, "offset": offset, "limit": limit, "returned": 0, "results": []}

    if start_time or end_time:
        df["updated"] = pd.to_datetime(df["updated"], errors="coerce")

        if start_time:
            df = df[df["updated"] >= pd.to_datetime(start_time, utc=True)]
        if end_time:
            df = df[df["updated"] <= pd.to_datetime(end_time, utc=True)]

    if df.empty:
        return {"total": 0, "offset": offset, "limit": limit, "returned": 0, "results": []}

    ack = _load_ack_df()
    df = df[~df["alert_id"].isin(ack["alert_id"])] if not ack.empty else df

    if df.empty:
        return {"total": 0, "offset": offset, "limit": limit, "returned": 0, "results": []}

    df["_sev_rank"] = df["predicted_time_to_depletion_min"].apply(
        lambda m: 0 if m <= 45 else 1 if m <= 90 else 2
    )
    df["__impact_sort"] = df["expected_deficit"].fillna(0) * -1
    df = df.sort_values(["_sev_rank", "__impact_sort"])

    total = len(df)
    df = df.iloc[offset : offset + limit]

    cards = [_row_to_card(r) for _, r in df.iterrows()]

    return {"total": total, "offset": offset, "limit": limit, "returned": len(cards), "results": cards}


# ================================
# ENDPOINT: ACKNOWLEDGE
# ================================
@app.post("/alerts/acknowledge/{alert_id}")
def acknowledge(alert_id: str):
    _save_ack(alert_id, status="acknowledged", reason="", user="")

    decision_payload = _build_decision_payload(alert_id, "ACCEPTED", "", "")

    try:
        pas_result = pas.publish_topic_message(PAS_DECISION_TOPIC, decision_payload)
    except Exception as e:
        pas_result = {"error": str(e)}

    return {"ok": True, "alert_id": alert_id, "status": "acknowledged", "pas": pas_result}


# ================================
# ENDPOINT: REJECT
# ================================
class RejectPayload(BaseModel):
    reason: Optional[str] = None
    user: Optional[str] = None


@app.post("/alerts/reject/{alert_id}")
def reject(alert_id: str, payload: RejectPayload):
    _save_ack(
        alert_id=alert_id,
        status="rejected",
        reason=payload.reason or "",
        user=payload.user or "",
    )

    decision_payload = _build_decision_payload(alert_id, "REJECTED", payload.reason, payload.user)

    try:
        pas_result = pas.publish_topic_message(PAS_DECISION_TOPIC, decision_payload)
    except Exception as e:
        pas_result = {"error": str(e)}

    return {
        "ok": True,
        "alert_id": alert_id,
        "status": "rejected",
        "reason": payload.reason,
        "user": payload.user,
        "timestamp": _now_iso(),
        "pas": pas_result,
    }


# ================================
# CORS
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
