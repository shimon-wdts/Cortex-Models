from datetime import datetime, timezone
import json
import os
from typing import Any, List, Optional

import pandas as pd

from app.core.config import settings
from app.models.schemas import FillCard, Rationale


ACK_COLUMNS = ["alert_id", "ack_time", "status", "reason", "user"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_decision_payload(alert_id: str, status: str, reason: str, user: str):
    return {
        "decisionId": f"dec-{alert_id}-{int(datetime.now().timestamp())}",
        "alertId": alert_id,
        "decision": status.upper(),
        "reason": reason or "",
        "user": user or "",
        "decidedAt": now_iso(),
    }


def _load_alerts_df() -> pd.DataFrame:
    alerts_csv_path = os.path.join(
        settings.data_dir, settings.alerts_csv_name
    )
    if not os.path.exists(alerts_csv_path):
        return pd.DataFrame()
    df = pd.read_csv(alerts_csv_path)
    dupes = [c for c in df.columns if c.endswith(".1") or c.endswith(".2")]
    df = df.drop(columns=dupes, errors="ignore")
    return df


def _load_ack_df() -> pd.DataFrame:
    ack_file_path = os.path.join(
        settings.data_dir, settings.ack_file_name
    )
    if os.path.exists(ack_file_path):
        df = pd.read_csv(ack_file_path)
        for col in ACK_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[ACK_COLUMNS]
    return pd.DataFrame(columns=ACK_COLUMNS)


def save_ack(alert_id: str, status: str, reason: str = "", user: str = "") -> None:
    df = _load_ack_df()
    df.loc[len(df)] = [alert_id, now_iso(), status, reason, user]
    ack_file_path = os.path.join(
        settings.data_dir, settings.ack_file_name
    )
    df.to_csv(ack_file_path, index=False)


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


def _is_exportable_alert(row: pd.Series) -> bool:
    required = ["alert_id", "table_id", "updated", "predicted_time_to_depletion_min"]
    return all(str(row.get(k, "")).strip() != "" for k in required)


def _build_exportable_content(row: pd.Series, card: FillCard) -> dict:
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
        "predicted_time_to_depletion_min": row.get("predicted_time_to_depletion_min"),
        "expected_deficit": row.get("expected_deficit"),
    }


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
        x_labels = [f"Point {i + 1}" for i in range(n)]

    severity = _severity_from_ttd(row.get("predicted_time_to_depletion_min"))
    option_count = int(row.get("option_count") or len(types))

    card = FillCard(
        alert_id=str(row.get("alert_id")),
        table_id=str(row.get("table_id")),
        entity_label=f"TABLE {row.get('table_id')}",
        alert_type=str(row.get("alert_type") or "ChipDepletion"),
        payout_risk_label=payout_label,
        updated=str(row.get("updated") or now_iso()),
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

    card.isActionable = True
    card.isExportable = _is_exportable_alert(row)
    card.exportable_content = _build_exportable_content(row, card) if card.isExportable else None

    return card


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
