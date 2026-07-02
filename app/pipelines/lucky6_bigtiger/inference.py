from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from app.pipelines.lucky6_bigtiger.build_features import Lucky6FeatureRecord, Lucky6FeatureResult, STATIC_GLOBAL_FEATURES

if TYPE_CHECKING:
    from app.models.pipeline_contracts import RunContext


EPS = 1e-6


@dataclass
class ModelSpec:
    name: str
    target: str
    mode: str
    feature_set: str
    base_ev_feature: str
    model_path: Path
    booster: Any


def static_global_col(name: str) -> int:
    return 5 + STATIC_GLOBAL_FEATURES.index(name)


def present(names: Iterable[str], token0_global: Mapping[str, int]) -> list[str]:
    return [name for name in names if name in token0_global]


def feature_columns(
    feature_set: str,
    d_static: int,
    n_static_tokens: int,
    n_static_flat: int,
) -> tuple[np.ndarray, list[str]]:
    token0_global = {name: 5 + i for i, name in enumerate(STATIC_GLOBAL_FEATURES)}

    rank_cols: list[int] = []
    rank_names: list[str] = []
    base_names = ["remain_norm", "frac", "delta", "z_like", "rank_index"]
    for tok in range(n_static_tokens):
        for col, base in enumerate(base_names):
            rank_cols.append(tok * d_static + col)
            rank_names.append(f"rank{tok}_{base}")

    def token0(names: Iterable[str]) -> tuple[list[int], list[str]]:
        names_kept = present(names, token0_global)
        return [token0_global[name] for name in names_kept], [f"g_{name}" for name in names_kept]

    shoe_state = present(["cards_remaining_norm", "decks_remaining_norm", "pct_shoe_done", "hands_remaining_norm"], token0_global)
    densities = present(["six_density", "ten_density", "low_density", "high_density", "rank_entropy", "five_six_seven_density"], token0_global)
    counts = present(["l6_rc_lvl1", "l6_tc_lvl1", "l6_rc_lvl2", "l6_tc_lvl2"], token0_global)
    interactions = present(
        [
            "six_x_decksrem",
            "six_x_pctdone",
            "tc1_x_pctdone",
            "tc2_x_pctdone",
            "comb_ev_lucky6_x_pct",
            "comb_ev_big_tiger_x_pct",
            "six_density_x_pct",
            "low_card_density_x_pct",
            "five_six_seven_density_x_pct",
            "rank_entropy_x_pct",
            "draw_volatility_proxy",
            "banker_draw_support",
            "player_draw_support",
            "draw_support_gap",
        ],
        token0_global,
    )
    comb = present(
        [
            "comb_p_l6_2",
            "comb_p_l6_3",
            "comb_p_l6_any",
            "comb_p_big_tiger",
            "comb_ev_lucky6",
            "comb_ev_big_tiger",
            "player_comb_p6_2",
            "player_comb_p6_3",
            "player_comb_p6_any",
            "player_comb_ev6_proxy",
            "banker_player_p6_2_gap",
            "banker_player_p6_3_gap",
            "banker_player_p6_any_gap",
            "total_p6_2_richness",
            "total_p6_3_richness",
            "total_p6_any_richness",
            "comb_ev_lucky6_margin",
            "comb_ev_big_tiger_margin",
        ],
        token0_global,
    )
    ev_history = present(
        [
            "ev_l6_delta_1",
            "ev_l6_delta_3",
            "ev_l6_delta_5",
            "ev_l6_delta_10",
            "ev_bt_delta_1",
            "ev_bt_delta_3",
            "ev_bt_delta_5",
            "ev_bt_delta_10",
            "ev_l6_roll_mean_3",
            "ev_l6_roll_mean_5",
            "ev_l6_roll_mean_10",
            "ev_bt_roll_mean_3",
            "ev_bt_roll_mean_5",
            "ev_bt_roll_mean_10",
            "phase_early",
            "phase_mid",
            "phase_late",
        ],
        token0_global,
    )
    rolling = [name for name in STATIC_GLOBAL_FEATURES if "rate_" in name or "avg_cards" in name or name.startswith("delta_")]

    if feature_set == "comb_only":
        cols, names = token0(comb)
    elif feature_set == "comb_pen":
        cols, names = token0(shoe_state + comb)
    elif feature_set == "rank_comb":
        c, n = token0(comb)
        cols, names = rank_cols + c, rank_names + n
    elif feature_set == "core_no_comb":
        c, n = token0(shoe_state + densities + counts + interactions)
        cols, names = rank_cols + c, rank_names + n
    elif feature_set == "core_no_seq":
        c, n = token0(shoe_state + densities + counts + interactions + comb + ev_history)
        cols, names = rank_cols + c, rank_names + n
    elif feature_set == "static_no_roll":
        keep = [name for name in STATIC_GLOBAL_FEATURES if name not in set(rolling)]
        c, n = token0(keep)
        cols, names = rank_cols + c, rank_names + n
    elif feature_set == "static_no_seq":
        cols = list(range(n_static_flat))
        names = [f"static_{i}" for i in cols]
    elif feature_set == "full":
        cols = list(range(n_static_flat + 360))
        names = [f"feature_{i}" for i in cols]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return np.asarray(cols, dtype=np.int64), names


def normalize_feature_set(feature_set: str) -> str:
    return feature_set[:-8] if feature_set.endswith("_derived") else feature_set


def append_derived(base: np.ndarray, full_rows: np.ndarray) -> np.ndarray:
    p2 = full_rows[:, static_global_col("comb_p_l6_2")]
    p3 = full_rows[:, static_global_col("comb_p_l6_3")]
    pany = full_rows[:, static_global_col("comb_p_l6_any")]
    ev_l6 = full_rows[:, static_global_col("comb_ev_lucky6")]
    ev_bt = full_rows[:, static_global_col("comb_ev_big_tiger")]
    pct = full_rows[:, static_global_col("pct_shoe_done")]
    rem = full_rows[:, static_global_col("hands_remaining_norm")]
    six = full_rows[:, static_global_col("six_density")]
    tc1 = full_rows[:, static_global_col("l6_tc_lvl1")]
    tc2 = full_rows[:, static_global_col("l6_tc_lvl2")]
    derived = np.column_stack(
        [
            p2 - p3,
            p3 / np.clip(p2, EPS, None),
            p2 / np.clip(pany, EPS, None),
            p3 / np.clip(pany, EPS, None),
            ev_l6 * pct,
            ev_l6 * rem,
            ev_bt * pct,
            ev_bt * rem,
            p2 * pct,
            p3 * pct,
            p2 * rem,
            p3 * rem,
            six * p2,
            six * p3,
            tc1 * p2,
            tc2 * p3,
        ]
    ).astype(np.float32)
    return np.concatenate([base, derived], axis=1)


def model_matrix(full_row: np.ndarray, spec: ModelSpec) -> np.ndarray:
    d_static = 5 + len(STATIC_GLOBAL_FEATURES)
    n_static_tokens = 10
    n_static_flat = n_static_tokens * d_static
    base_set = normalize_feature_set(spec.feature_set)
    feat_idx, _ = feature_columns(base_set, d_static, n_static_tokens, n_static_flat)

    if spec.mode == "residual" and "no_comb" in base_set:
        comb_cols = {static_global_col(name) for name in STATIC_GLOBAL_FEATURES if name.startswith("comb_")}
        keep = [idx for idx, col in enumerate(feat_idx.tolist()) if col not in comb_cols]
        feat_idx = feat_idx[keep]

    out = np.asarray(full_row[:, feat_idx], dtype=np.float32, order="C")
    if spec.feature_set.endswith("_derived"):
        out = append_derived(out, full_row)
    return out


def base_ev(full_row: np.ndarray, spec: ModelSpec) -> float:
    return float(full_row[0, static_global_col(spec.base_ev_feature)])


def load_models(model_dir: Path, model_set: str) -> dict[str, ModelSpec]:
    import lightgbm as lgb

    manifest_path = model_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if model_set not in manifest:
        raise ValueError(f"model_set must be one of {sorted(manifest)}, got {model_set!r}")

    specs: dict[str, ModelSpec] = {}
    for name, item in manifest[model_set].items():
        model_path = model_dir / item["model_file"]
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        specs[name] = ModelSpec(
            name=name,
            target=str(item["target"]),
            mode=str(item["mode"]),
            feature_set=str(item["feature_set"]),
            base_ev_feature=str(item["base_ev_feature"]),
            model_path=model_path,
            booster=lgb.Booster(model_file=str(model_path)),
        )
    return specs


def score_models(full_row: np.ndarray, models: Mapping[str, ModelSpec]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, spec in models.items():
        x_model = model_matrix(full_row, spec)
        pred = float(spec.booster.predict(x_model)[0])
        score = base_ev(full_row, spec) + pred if spec.mode == "residual" else pred
        scores[name] = float(score)
    return scores


def score_feature_records(
    features: Lucky6FeatureResult,
    *,
    model_dir: Path,
    model_set: str,
    context: "RunContext",
    json_limit: int | None = None,
) -> list[dict[str, Any]]:
    models = load_models(model_dir, model_set)
    created_at = datetime.now(UTC).isoformat()
    outputs: list[dict[str, Any]] = []

    for record in features.records:
        if record.feature_row is None:
            continue
        scores = score_models(record.feature_row, models)
        outputs.append(_prediction_payload(record, scores, context, model_set, created_at))
        if json_limit and json_limit > 0 and len(outputs) >= json_limit:
            break

    return outputs


def _prediction_payload(
    record: Lucky6FeatureRecord,
    scores: Mapping[str, float],
    context: "RunContext",
    model_set: str,
    created_at: str,
) -> dict[str, Any]:
    lucky6adv = round(float(scores["lucky6"]), 8)
    big_tiger_adv = round(float(scores["big_tiger"]), 8)
    return {
        "insights_id": f"l6bt_{uuid4().hex[:17]}",
        "model_name": context.model_name,
        "run_id": context.run_id,
        "model_set": model_set,
        "created_at": created_at,
        "event_ts": _timestamp_to_json(record.event_ts),
        "game_start_ts": _timestamp_to_json(record.game_start_ts),
        "gaming_day": record.gaming_day or None,
        "hand_id": record.hand_id,
        "shoe_id": record.shoe_id,
        "game_id": record.game_id,
        "table_id": record.table_id or None,
        "table_name": record.table_name or None,
        "pit_name": record.pit_name or None,
        "gaming_area": record.gaming_area or None,
        "lucky6adv": lucky6adv,
        "bigTigeradv": big_tiger_adv,
        "score_status": "scored",
        "cards_remaining": record.cards_remaining,
        "decks_remaining": record.decks_remaining,
        "history_size": record.history_size,
        "model": {
            "type": "ShoeAdvantage",
            "version": str(context.parameters.get("model_version", model_set)),
            "feature_version": str(context.parameters.get("feature_version", "1.0")),
            "run_id": context.run_id,
        },
        "payload": {
            "result": {
                "lucky6adv": lucky6adv,
                "bigTigeradv": big_tiger_adv,
            },
            "source": {
                "table": "t_game",
                "game_id": record.game_id,
                "shoe_id": record.shoe_id,
                "hand_id": record.hand_id,
            },
        },
    }


def _timestamp_to_json(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.isoformat()
