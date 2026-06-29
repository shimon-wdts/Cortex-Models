from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "tray_balance", "available_for_payout", "safety_reserve",
    "OUT_total_60m", "bettor_out_60m", "chip_out_60m", "out_rate_per_min",
    "expected_payout_next30", "expected_payout_next60",
    "net_buffer_next30", "net_buffer_next60",
    "tray_age_hours", "minutes_since_last_seen", "tray_stale_flag",
    "expected_deficit_next30", "expected_deficit_next60",
    "denom_risk_next30_rule", "denom_risk_next60_rule",
]


def default_model_dir() -> Path:
    return Path(__file__).resolve().parent / "models" / "current_model"


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def rule_score(df: pd.DataFrame) -> pd.Series:
    deficit60 = numeric_series(df, "expected_deficit_next60").to_numpy()
    buffer30 = numeric_series(df, "net_buffer_next30").to_numpy()
    out_rate = numeric_series(df, "out_rate_per_min").to_numpy()
    denom60 = numeric_series(df, "denom_risk_next60_rule").to_numpy()
    scale = max(np.nanmedian(np.abs(buffer30)) if np.isfinite(buffer30).any() else 1.0, 1.0)
    rate_scale = max(np.nanmedian(np.abs(out_rate)) if np.isfinite(out_rate).any() else 1.0, 1e-6)
    score = (2.0 * (deficit60 / scale)) + (1.2 * (-buffer30 / scale)) + (0.8 * (out_rate / rate_scale))
    prob = sigmoid(score)
    prob = np.where(denom60.astype(int) == 1, np.maximum(prob, 0.95), prob)
    return pd.Series(np.clip(prob, 0.0, 1.0), index=df.index, name="need_prob")


def load_model_artifacts(model_dir: Path) -> tuple[dict, object | None]:
    meta_path = model_dir / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing model metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = model_dir / "model.pkl"
    estimator = None
    if model_path.exists():
        try:
            with model_path.open("rb") as fh:
                estimator = pickle.load(fh)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency while loading saved model. Install dependencies from requirements.txt."
            ) from exc
    return metadata, estimator


def score_with_saved_model(features: pd.DataFrame, model_dir: Path | None = None, threshold: float = 0.60) -> pd.DataFrame:
    resolved_model_dir = model_dir or default_model_dir()
    out = features.copy()
    if out.empty:
        meta_path = resolved_model_dir / "model_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing model metadata: {meta_path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        out["need_prob"] = pd.Series(dtype=float)
        out["need_pred"] = pd.Series(dtype=int)
        out["decision_threshold"] = pd.Series(dtype=float)
        out["model_type"] = str(metadata.get("model_type", "unknown"))
        out["recommended_action"] = pd.Series(dtype=str)
        out["risk_band"] = pd.Series(dtype=str)
        out["insight_summary"] = pd.Series(dtype=str)
        return out
    metadata, estimator = load_model_artifacts(resolved_model_dir)
    feature_cols = metadata.get("feature_cols", [])
    if estimator is not None and feature_cols:
        X = out.reindex(columns=feature_cols, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["need_prob"] = estimator.predict_proba(X)[:, 1]
    else:
        out["need_prob"] = rule_score(out)
    out["need_pred"] = (out["need_prob"] >= threshold).astype(int)
    out["decision_threshold"] = float(threshold)
    out["model_type"] = str(metadata.get("model_type", "unknown"))
    out["recommended_action"] = out["need_pred"].map({1: "DISPATCH_FILL", 0: "NO_ACTION"})
    out["risk_band"] = pd.cut(
        out["need_prob"],
        bins=[-0.001, 0.30, 0.60, 0.85, 1.0],
        labels=["low", "watch", "high", "critical"],
    ).astype(str)
    out["insight_summary"] = out.apply(
        lambda row: f"table {row.get('table_id', 'unknown')} probability={float(row.get('need_prob', 0.0)):.3f} action={row.get('recommended_action', 'UNKNOWN')}",
        axis=1,
    )
    return out


def json_safe_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def severity_from_row(row: pd.Series) -> str:
    prob = float(pd.to_numeric(row.get("need_prob", 0.0), errors="coerce") or 0.0)
    if prob >= 0.85:
        return "Critical"
    if prob >= 0.60:
        return "High"
    if prob >= 0.30:
        return "Medium"
    return "Low"


def build_recommendations(row: pd.Series) -> tuple[list[str], list[str], list[float], list[int], list[float], list[str], list[dict[str, object]]]:
    need_pred = int(pd.to_numeric(row.get("need_pred", 0), errors="coerce") or 0)
    need_prob = float(pd.to_numeric(row.get("need_prob", 0.0), errors="coerce") or 0.0)
    time_to_depletion = int(min(999, max(0, round((1.0 - min(need_prob, 0.999)) * 240)))) if need_pred else 999
    impact = round(need_prob * 100.0, 2)
    roi = round(max(0.0, need_prob * 10.0), 5)

    if need_pred:
        types = ["cage_fill", "table_transfer"]
        recommendations = ["Dispatch fill from cage", "Consider chip transfer between tables"]
    else:
        types = ["monitor"]
        recommendations = ["Monitor table condition"]

    modeled_impact = [impact] if len(types) == 1 else [impact, round(impact * 0.8, 2)]
    time_to_depletion_list = [time_to_depletion] * len(types)
    roi_list = [roi] if len(types) == 1 else [roi, round(roi * 0.9, 5)]
    roi_unit = ["x"] * len(types)

    rationale = []
    for idx in range(len(types)):
        pretty = (
            f"Balance is {round(float(pd.to_numeric(row.get('available_for_payout', 0.0), errors='coerce') or 0.0), 2):,.0f} above buffer | "
            f"Expected payout next 30 min: {round(float(pd.to_numeric(row.get('expected_payout_next30', 0.0), errors='coerce') or 0.0), 2):,.0f} | "
            f"EV/hr impact: {round(modeled_impact[idx], 2):,.0f} | "
            f"Safety buffer: {round(float(pd.to_numeric(row.get('safety_reserve', 0.0), errors='coerce') or 0.0), 2):,.0f}"
        )
        rationale.append(
            {
                "rel_balance": round(float(pd.to_numeric(row.get("available_for_payout", 0.0), errors="coerce") or 0.0), 2),
                "exp_payout_next30": round(float(pd.to_numeric(row.get("expected_payout_next30", 0.0), errors="coerce") or 0.0), 2),
                "exp_payout_hr": round(modeled_impact[idx], 2),
                "safety_buffer": round(float(pd.to_numeric(row.get("safety_reserve", 0.0), errors="coerce") or 0.0), 2),
                "model": row.get("model_type"),
                "pretty": pretty,
            }
        )
    return types, recommendations, modeled_impact, time_to_depletion_list, roi_list, roi_unit, rationale


def build_fill_alerts_json(scored: pd.DataFrame, limit: int | None = None) -> list[dict[str, object]]:
    work = scored.copy()
    if "snapshot_ts" in work.columns:
        work["snapshot_ts"] = pd.to_datetime(work["snapshot_ts"], errors="coerce", utc=True)
    work = work.sort_values(["need_prob", "snapshot_ts"], ascending=[False, False], na_position="last")
    if limit is not None:
        work = work.head(limit)

    alerts: list[dict[str, object]] = []
    for _, row in work.iterrows():
        table_id = str(row.get("table_id", "unknown"))
        snapshot_ts = pd.to_datetime(row.get("snapshot_ts"), errors="coerce", utc=True)
        snapshot_text = snapshot_ts.strftime("%Y-%m-%dT%H:%M") if not pd.isna(snapshot_ts) else "unknown"
        alert_id = f"{table_id}_{snapshot_text}"
        severity = severity_from_row(row)
        types, recommendations, modeled_impact, time_to_depletion_list, roi_list, roi_unit, rationale = build_recommendations(row)
        need_prob = float(pd.to_numeric(row.get("need_prob", 0.0), errors="coerce") or 0.0)
        chart_points = [
            round(float(pd.to_numeric(row.get("tray_balance", 0.0), errors="coerce") or 0.0), 2),
            round(float(pd.to_numeric(row.get("available_for_payout", 0.0), errors="coerce") or 0.0), 2),
            round(float(pd.to_numeric(row.get("expected_payout_next30", 0.0), errors="coerce") or 0.0), 2),
            round(float(pd.to_numeric(row.get("expected_payout_next60", 0.0), errors="coerce") or 0.0), 2),
            round(float(pd.to_numeric(row.get("out_rate_per_min", 0.0), errors="coerce") or 0.0), 2),
        ]
        chart_series = [[chart_points, chart_points] for _ in types]
        updated = snapshot_ts.isoformat() if not pd.isna(snapshot_ts) else pd.Timestamp.utcnow().isoformat()
        threshold = float(pd.to_numeric(row.get("decision_threshold", 0.6), errors="coerce") or 0.6)

        alert = {
            "alert_id": alert_id,
            "table_id": table_id,
            "entity_label": f"TABLE {table_id}",
            "alert_type": "ChipDepletion",
            "cohort": str(row.get("risk_band", severity.lower())),
            "payout_risk_label": "Below buffer" if need_prob >= threshold else "Above buffer",
            "updated": updated,
            "severity": severity,
            "option_count": len(types),
            "types": types,
            "modeled_impact_per_hr": modeled_impact,
            "time_to_depletion_min": time_to_depletion_list,
            "roi": roi_list,
            "roi_unit": roi_unit,
            "trigger_metric": str(row.get("risk_band", "need_probability")),
            "recommendation": recommendations,
            "chart_series": chart_series,
            "chart_labels": ["EV/hr (recommended)", "EV/hr (baseline)"],
            "x_labels": ["Tray Balance", "Avail Payout", "Exp 30m", "Exp 60m", "Out Rate"],
            "chart_label": "EV/hr",
            "rationale": rationale,
            "isExportable": True,
            "exportable_content": {
                "alert_id": alert_id,
                "table_id": table_id,
                "entity_label": f"TABLE {table_id}",
                "alert_type": "ChipDepletion",
                "severity": severity,
                "updated": updated,
                "recommended_type": types[0] if types else None,
                "recommended_time_to_depletion_min": time_to_depletion_list[0] if time_to_depletion_list else None,
                "recommended_modeled_impact_per_hr": modeled_impact[0] if modeled_impact else None,
                "recommended_roi": roi_list[0] if roi_list else None,
                "recommended_roi_unit": roi_unit[0] if roi_unit else None,
                "recommended_recommendation": recommendations[0] if recommendations else None,
                "recommended_rationale": rationale[0]["pretty"] if rationale else None,
                "expected_deficit": round(float(pd.to_numeric(row.get("expected_deficit_next60", 0.0), errors="coerce") or 0.0), 2),
            },
            "isActionable": bool(int(pd.to_numeric(row.get("need_pred", 0), errors="coerce") or 0)),
        }
        alerts.append({k: json_safe_value(v) if not isinstance(v, (list, dict)) else v for k, v in alert.items()})
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a feature CSV with the packaged saved model.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-dir", default=str(default_model_dir()))
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--json-limit", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs_pipeline/inference")
    args = parser.parse_args()

    features = pd.read_csv(Path(args.features), low_memory=False)
    scored = score_with_saved_model(features, Path(args.model_dir), threshold=args.threshold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "scored_tables_latest.csv", index=False)
    insight_cols = [c for c in [
        "snapshot_ts", "table_id", "need_prob", "need_pred", "decision_threshold",
        "risk_band", "recommended_action", "insight_summary",
    ] if c in scored.columns]
    scored[insight_cols].sort_values("need_prob", ascending=False).to_csv(output_dir / "predicted_fill_insights_latest.csv", index=False)
    (output_dir / "fill_alerts_latest.json").write_text(json.dumps(build_fill_alerts_json(scored, args.json_limit), indent=2), encoding="utf-8")
    print(f"Scored rows: {len(scored)}")
    print(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()
