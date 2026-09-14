from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.pipelines.predicted_fills.fetch_data import FeatureRepository, RawSources


@dataclass(frozen=True)
class FeatureConfig:
    window_min: int = 60
    horizon30_min: int = 30
    horizon60_min: int = 60
    snapshot_minutes: int = 15
    safety_reserve_pct: float = 0.15
    warn_buffer_pct: float = 0.10
    severe_buffer_pct: float = 0.05
    stale_tray_hours: float = 24.0
    enable_tray_freshness_gate: bool = True


@dataclass
class FeatureResult:
    features: pd.DataFrame
    denom_risk: pd.DataFrame

    def __len__(self) -> int:
        return len(self.features) + len(self.denom_risk)


@dataclass
class SnapshotResult:
    base_features: pd.DataFrame
    denom_risk: pd.DataFrame


def safe_fill_numeric(df: pd.DataFrame, value: float = 0.0) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    cols = out.select_dtypes(include=[np.number]).columns
    out[cols] = out[cols].fillna(value)
    return out


def build_feature_dataset(
    sources: RawSources,
    topology: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    config: FeatureConfig | None = None,
) -> FeatureResult:
    cfg = config or FeatureConfig()
    builder = FeatureBuilder(cfg)
    repository = FeatureRepository()
    anchors = pd.date_range(start=start_ts, end=end_ts, freq=f"{cfg.snapshot_minutes}min", tz="UTC")

    feature_frames: list[pd.DataFrame] = []
    denom_frames: list[pd.DataFrame] = []
    for anchor_ts in anchors:
        snapshot = builder.build_snapshot(repository, sources, pd.Timestamp(anchor_ts))
        feature_frames.append(snapshot.base_features)
        if not snapshot.denom_risk.empty:
            denom = snapshot.denom_risk.copy()
            denom["snapshot_ts"] = pd.Timestamp(anchor_ts)
            denom_frames.append(denom)

    features = pd.concat([f for f in feature_frames if f is not None and not f.empty], ignore_index=True) if feature_frames else pd.DataFrame()
    denom_risk = pd.concat([d for d in denom_frames if d is not None and not d.empty], ignore_index=True) if denom_frames else pd.DataFrame()
    features = add_table_labels(features, topology, sources.tray_scans)
    return FeatureResult(features=features, denom_risk=denom_risk)


class FeatureBuilder:
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def build_snapshot(
        self,
        repository: FeatureRepository,
        sources: RawSources,
        anchor_ts: pd.Timestamp,
    ) -> SnapshotResult:
        tray = self.latest_tray_balance(sources.tray_scans, anchor_ts)
        chip_mv = self.chip_movement_window(sources.chip_updates, anchor_ts)
        tbet = self.bet_flow_window(sources.bets, anchor_ts)
        denom_inv = self.denom_inventory(sources.chip_inventory, anchor_ts)
        denom_mix = self.denom_mix_window(repository, sources.chip_updates, sources.chip_inventory, anchor_ts)
        if denom_mix.empty and not denom_inv.empty:
            denom_mix = denom_inv[["table_id", "denom", "value"]].rename(columns={"value": "mix_weight"})
        base = self.build_state_features(chip_mv, tbet, tray, anchor_ts)
        denom_risk = self.compute_denom_risk(denom_inv, base, denom_mix)
        enriched = self.compute_risk_features(base, denom_risk)
        enriched["snapshot_ts"] = pd.Timestamp(anchor_ts)
        return SnapshotResult(base_features=enriched, denom_risk=denom_risk)

    def latest_tray_balance(self, tray_scans: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.DataFrame:
        schema = ["table_id", "tray_balance", "tray_ts", "tray_source"]
        if tray_scans is None or tray_scans.empty:
            return pd.DataFrame(columns=schema)
        df = tray_scans[tray_scans["tray_ts"] <= anchor_ts].copy()
        if df.empty:
            return pd.DataFrame(columns=schema)
        latest = df.sort_values(["table_id", "tray_ts"]).groupby("table_id", as_index=False).tail(1)
        out = latest[["table_id", "tray_balance", "tray_ts"]].copy()
        out["tray_source"] = "t_cage_chip_tray_scan_rwlv"
        return out[schema]

    def chip_movement_window(self, chip_updates: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.DataFrame:
        schema = ["table_id", "chip_in_60m", "chip_out_60m", "last_activity_ts"]
        if chip_updates is None or chip_updates.empty:
            return pd.DataFrame(columns=schema)
        w_start = anchor_ts - pd.Timedelta(minutes=self.config.window_min)
        df = chip_updates[(chip_updates["update_ts"] >= w_start) & (chip_updates["update_ts"] <= anchor_ts)].copy()
        df = df[df["update_type"].isin(["CHIPS_OUT", "CHIPS_IN"])].copy()
        if df.empty:
            return pd.DataFrame(columns=schema)
        is_out = df["update_type"].eq("CHIPS_OUT")
        is_in = df["update_type"].eq("CHIPS_IN")
        tmp = pd.DataFrame({
            "table_id": np.where(is_out.to_numpy(), df["from_table_id"].to_numpy(), df["to_table_id"].to_numpy()),
            "_ts": df["update_ts"].to_numpy(),
            "chip_out": np.where(is_out.to_numpy(), df["update_value"].to_numpy(), 0.0),
            "chip_in": np.where(is_in.to_numpy(), df["update_value"].to_numpy(), 0.0),
        })
        tmp["table_id"] = pd.to_numeric(tmp["table_id"], errors="coerce")
        tmp = tmp[tmp["table_id"].notna()].copy()
        if tmp.empty:
            return pd.DataFrame(columns=schema)
        tmp["table_id"] = tmp["table_id"].astype(int)
        grouped = tmp.groupby("table_id", as_index=False).agg(
            chip_in_60m=("chip_in", "sum"),
            chip_out_60m=("chip_out", "sum"),
            last_activity_ts=("_ts", "max"),
        )
        return grouped[schema]

    def bet_flow_window(self, bets: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.DataFrame:
        schema = ["table_id", "bettor_in_60m", "bettor_out_60m", "count_bets_60m", "last_bet_ts", "last_payout_ts"]
        if bets is None or bets.empty:
            return pd.DataFrame(columns=schema)
        w_start = anchor_ts - pd.Timedelta(minutes=self.config.window_min)
        df = bets[(bets["payout_ts"] >= w_start) & (bets["payout_ts"] <= anchor_ts)].copy()
        if df.empty:
            return pd.DataFrame(columns=schema)
        win = pd.to_numeric(df["casino_win"], errors="coerce").fillna(0.0)
        loss = pd.to_numeric(df["casino_loss"], errors="coerce").fillna(0.0)
        bettor_in = np.where(win > 0, win, 0.0)
        bettor_out = np.where(loss > 0, loss, np.where(win < 0, -win, 0.0))
        tmp = pd.DataFrame({
            "table_id": df["table_id"].astype(int),
            "bettor_in": bettor_in,
            "bettor_out": bettor_out,
            "_ts": df["payout_ts"],
            "_is_payout": bettor_out > 0,
        })
        grouped = tmp.groupby("table_id", as_index=False).agg(
            bettor_in_60m=("bettor_in", "sum"),
            bettor_out_60m=("bettor_out", "sum"),
            count_bets_60m=("table_id", "size"),
            last_bet_ts=("_ts", "max"),
        )
        payout_ts = tmp.loc[tmp["_is_payout"], ["table_id", "_ts"]].groupby("table_id", as_index=False)["_ts"].max().rename(columns={"_ts": "last_payout_ts"})
        return grouped.merge(payout_ts, on="table_id", how="left")[schema]

    def denom_inventory(self, chip_inventory: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.DataFrame:
        schema = ["table_id", "denom", "qty", "value", "inv_ts", "inv_source"]
        if chip_inventory is None or chip_inventory.empty:
            return pd.DataFrame(columns=schema)
        df = chip_inventory.copy()
        if df["updated_ts"].notna().any():
            df = df[(df["updated_ts"].isna()) | (df["updated_ts"] <= anchor_ts)].copy()
        if df.empty:
            return pd.DataFrame(columns=schema)
        grouped = df.groupby(["table_id", "denom"], as_index=False).agg(qty=("chip_id", "count"), inv_ts=("updated_ts", "max"))
        grouped["value"] = grouped["qty"] * grouped["denom"]
        grouped["inv_source"] = "t_chip_rwlv"
        return grouped[schema]

    def denom_mix_window(
        self,
        repository: FeatureRepository,
        chip_updates: pd.DataFrame,
        chip_inventory: pd.DataFrame,
        anchor_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        schema = ["table_id", "denom", "mix_weight"]
        if chip_updates is None or chip_updates.empty or chip_inventory is None or chip_inventory.empty:
            return pd.DataFrame(columns=schema)
        w_start = anchor_ts - pd.Timedelta(minutes=self.config.window_min)
        updates = chip_updates[(chip_updates["update_ts"] >= w_start) & (chip_updates["update_ts"] <= anchor_ts) & (chip_updates["update_type"] == "CHIPS_OUT")].copy()
        if updates.empty:
            return pd.DataFrame(columns=schema)
        expanded = repository.expand_chip_ids(updates)
        if expanded.empty:
            return pd.DataFrame(columns=schema)
        chips = chip_inventory[["chip_id", "denom"]].copy()
        chips["denom"] = pd.to_numeric(chips["denom"], errors="coerce")
        chips = chips[chips["denom"].notna()].copy()
        merged = expanded.merge(chips, on="chip_id", how="left")
        merged = merged[merged["denom"].notna()].copy()
        if merged.empty:
            return pd.DataFrame(columns=schema)
        merged["mix_weight"] = merged["denom"].astype(float)
        return merged.groupby(["table_id", "denom"], as_index=False)["mix_weight"].sum()

    def build_state_features(self, chip_mv: pd.DataFrame, tbet: pd.DataFrame, tray: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.DataFrame:
        base = tray.merge(chip_mv, on="table_id", how="outer").merge(tbet, on="table_id", how="outer")
        base = safe_fill_numeric(base, 0.0)
        base["OUT_total_60m"] = base["chip_out_60m"] + base["bettor_out_60m"]
        base["IN_total_60m"] = base["chip_in_60m"] + base["bettor_in_60m"]
        base["out_rate_per_min"] = base["OUT_total_60m"] / float(self.config.window_min)
        base["expected_payout_next30"] = base["out_rate_per_min"] * float(self.config.horizon30_min)
        base["expected_payout_next60"] = base["out_rate_per_min"] * float(self.config.horizon60_min)
        base["tray_ts"] = pd.to_datetime(base.get("tray_ts"), errors="coerce", utc=True)
        base["tray_age_hours"] = (anchor_ts - base["tray_ts"]) / pd.Timedelta(hours=1)
        base.loc[base["tray_age_hours"] < 0, "tray_age_hours"] = 0.0
        base["safety_reserve"] = base["tray_balance"] * self.config.safety_reserve_pct
        base["available_for_payout"] = base["tray_balance"] - base["safety_reserve"]
        base["net_buffer_next30"] = base["available_for_payout"] - base["expected_payout_next30"]
        base["net_buffer_next60"] = base["available_for_payout"] - base["expected_payout_next60"]
        if self.config.enable_tray_freshness_gate:
            base["tray_stale_flag"] = ((base["tray_ts"].notna()) & (base["tray_age_hours"] > self.config.stale_tray_hours)).astype(int)
        else:
            base["tray_stale_flag"] = 0
        base["last_activity_ts"] = pd.to_datetime(base.get("last_activity_ts"), errors="coerce", utc=True)
        base["last_bet_ts"] = pd.to_datetime(base.get("last_bet_ts"), errors="coerce", utc=True)
        base["last_seen_ts"] = base[["tray_ts", "last_activity_ts", "last_bet_ts"]].max(axis=1)
        base["minutes_since_last_seen"] = (anchor_ts - base["last_seen_ts"]) / np.timedelta64(1, "m")
        return base

    def compute_denom_risk(self, inv: pd.DataFrame, base: pd.DataFrame, denom_mix: pd.DataFrame) -> pd.DataFrame:
        schema = [
            "table_id", "denom", "qty", "value", "inv_ts", "inv_source",
            "projected_value_next30", "projected_qty_next30", "qty_after_next30", "denom_deplete_next30_flag",
            "projected_value_next60", "projected_qty_next60", "qty_after_next60", "denom_deplete_next60_flag",
            "minutes_to_zero_est",
        ]
        if inv is None or inv.empty or base is None or base.empty:
            return pd.DataFrame(columns=schema)
        need = base[["table_id", "expected_payout_next30", "expected_payout_next60", "OUT_total_60m"]].copy()
        if denom_mix is None or denom_mix.empty:
            denom_mix = inv[["table_id", "denom"]].drop_duplicates()
            denom_mix["mix_weight"] = 1.0
        denom_mix = denom_mix.copy()
        denom_mix["mix_weight"] = pd.to_numeric(denom_mix["mix_weight"], errors="coerce").fillna(0.0)
        denom_mix = denom_mix[denom_mix["mix_weight"] > 0].copy()
        if denom_mix.empty:
            return pd.DataFrame(columns=schema)
        wsum = denom_mix.groupby("table_id", as_index=False)["mix_weight"].sum().rename(columns={"mix_weight": "mix_sum"})
        denom_mix = denom_mix.merge(wsum, on="table_id", how="left")
        denom_mix["mix_frac"] = np.where(denom_mix["mix_sum"] > 0, denom_mix["mix_weight"] / denom_mix["mix_sum"], 0.0)
        df = inv.merge(denom_mix[["table_id", "denom", "mix_frac"]], on=["table_id", "denom"], how="left").merge(need, on="table_id", how="left")
        df = safe_fill_numeric(df, 0.0)
        df["projected_value_next30"] = df["expected_payout_next30"] * df["mix_frac"]
        df["projected_qty_next30"] = np.where(df["denom"] > 0, df["projected_value_next30"] / df["denom"], 0.0)
        df["qty_after_next30"] = df["qty"] - df["projected_qty_next30"]
        df["denom_deplete_next30_flag"] = (df["qty_after_next30"] < 0).astype(int)
        df["projected_value_next60"] = df["expected_payout_next60"] * df["mix_frac"]
        df["projected_qty_next60"] = np.where(df["denom"] > 0, df["projected_value_next60"] / df["denom"], 0.0)
        df["qty_after_next60"] = df["qty"] - df["projected_qty_next60"]
        df["denom_deplete_next60_flag"] = (df["qty_after_next60"] < 0).astype(int)
        df["out_per_min_value"] = (df["OUT_total_60m"] * df["mix_frac"]) / float(self.config.window_min)
        df["minutes_to_zero_est"] = np.where(df["out_per_min_value"] > 0, df["value"] / df["out_per_min_value"], np.nan)
        return df[schema]

    def compute_risk_features(self, base: pd.DataFrame, denom_risk: pd.DataFrame) -> pd.DataFrame:
        out = base.copy()
        out["warn_buffer_value"] = out["tray_balance"] * self.config.warn_buffer_pct
        out["severe_buffer_value"] = out["tray_balance"] * self.config.severe_buffer_pct
        out["expected_deficit_next30"] = (out["expected_payout_next30"] - out["available_for_payout"]).clip(lower=0.0)
        out["expected_deficit_next60"] = (out["expected_payout_next60"] - out["available_for_payout"]).clip(lower=0.0)

        if denom_risk is not None and not denom_risk.empty:
            worst30 = (
                denom_risk.sort_values(["denom_deplete_next30_flag", "qty_after_next30"], ascending=[False, True])
                .groupby("table_id", as_index=False)
                .head(1)[["table_id", "denom", "qty_after_next30", "denom_deplete_next30_flag", "minutes_to_zero_est"]]
                .rename(columns={
                    "denom": "worst_denom_next30",
                    "qty_after_next30": "worst_denom_qty_after_next30",
                    "denom_deplete_next30_flag": "denom_risk_next30_rule",
                    "minutes_to_zero_est": "worst_denom_minutes_to_zero",
                })
            )
            worst60 = (
                denom_risk.sort_values(["denom_deplete_next60_flag", "qty_after_next60"], ascending=[False, True])
                .groupby("table_id", as_index=False)
                .head(1)[["table_id", "denom", "qty_after_next60", "denom_deplete_next60_flag"]]
                .rename(columns={
                    "denom": "worst_denom_next60",
                    "qty_after_next60": "worst_denom_qty_after_next60",
                    "denom_deplete_next60_flag": "denom_risk_next60_rule",
                })
            )
            out = out.merge(worst30, on="table_id", how="left").merge(worst60, on="table_id", how="left")
        else:
            out["denom_risk_next30_rule"] = 0
            out["denom_risk_next60_rule"] = 0

        out["denom_risk_next30_rule"] = pd.to_numeric(out.get("denom_risk_next30_rule", 0), errors="coerce").fillna(0).astype(int)
        out["denom_risk_next60_rule"] = pd.to_numeric(out.get("denom_risk_next60_rule", 0), errors="coerce").fillna(0).astype(int)
        out["needs_fill_now_rule"] = (
            (out["available_for_payout"] <= out["severe_buffer_value"])
            | (out["expected_deficit_next30"] > 0)
            | (out["denom_risk_next30_rule"] == 1)
            | (pd.to_numeric(out.get("worst_denom_minutes_to_zero", pd.Series(9999, index=out.index)), errors="coerce").fillna(9999) <= 30)
        ).astype(int)
        out["needs_fill_next30_rule"] = (
            (out["expected_deficit_next30"] > 0)
            | (out["net_buffer_next30"] <= out["warn_buffer_value"])
            | (out["denom_risk_next30_rule"] == 1)
        ).astype(int)
        out["needs_fill_next60_rule"] = (
            (out["expected_deficit_next60"] > 0)
            | (out["net_buffer_next60"] <= out["warn_buffer_value"])
            | (out["denom_risk_next60_rule"] == 1)
            | (pd.to_numeric(out.get("worst_denom_minutes_to_zero", pd.Series(9999, index=out.index)), errors="coerce").fillna(9999) <= 60)
        ).astype(int)
        out["true_need_severity"] = "low"
        out.loc[out["needs_fill_next60_rule"] == 1, "true_need_severity"] = "medium"
        out.loc[out["needs_fill_next30_rule"] == 1, "true_need_severity"] = "high"
        out.loc[out["needs_fill_now_rule"] == 1, "true_need_severity"] = "critical"
        return out

def add_table_labels(features: pd.DataFrame, topology: pd.DataFrame, tray_scans: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    labels = pd.DataFrame(columns=["table_id", "table_name", "pit_name", "gaming_area"])
    if tray_scans is not None and not tray_scans.empty:
        latest = tray_scans.sort_values(["table_id", "tray_ts"]).groupby("table_id", as_index=False).tail(1)
        cols = [c for c in ["table_id", "table_name", "pit_name", "gaming_area"] if c in latest.columns]
        labels = latest[cols].copy()
    if topology is not None and not topology.empty:
        topology = topology.copy()
        topology["table_id"] = pd.to_numeric(topology["table_id"], errors="coerce")
        topology = topology[topology["table_id"].notna()].copy()
        topology["table_id"] = topology["table_id"].astype(int)
        out = out.merge(topology, on="table_id", how="left")
    if not labels.empty:
        out = out.merge(labels, on="table_id", how="left")
    return out


def add_quality_flags(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["has_real_tray_scan"] = out["tray_balance"].notna() & out["tray_ts"].notna()
    out["fresh_tray_scan"] = out["has_real_tray_scan"] & (pd.to_numeric(out.get("tray_age_hours"), errors="coerce").fillna(9999) <= 24.0)
    out["score_usable_for_shadow_review"] = out["fresh_tray_scan"]
    out["score_usable_for_auto_dispatch"] = False
    out["data_quality_status"] = "RWLV_REAL_TRAY_SCAN_FRESH"
    out.loc[~out["has_real_tray_scan"], "data_quality_status"] = "RWLV_MISSING_TRAY_SCAN"
    out.loc[out["has_real_tray_scan"] & ~out["fresh_tray_scan"], "data_quality_status"] = "RWLV_STALE_TRAY_SCAN"
    return out
