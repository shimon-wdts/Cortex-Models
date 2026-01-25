from pydantic import BaseModel
from typing import List, Optional, Any


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

    isExportable: bool = False
    exportable_content: Optional[Any] = None
    isActionable: bool = False


class RejectPayload(BaseModel):
    reason: Optional[str] = None
    user: Optional[str] = None
