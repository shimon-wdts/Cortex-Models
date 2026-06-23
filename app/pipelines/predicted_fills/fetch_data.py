from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

SOURCE_TABLES = {
    "tray_scans": "t_cage_chip_tray_scan_rwlv",
    "chip_inventory": "t_chip_rwlv",
    "chip_updates": "t_chip_txn_update_rwlv",
    "bets": "t_bet_rwlv",
    "topology": "t_topology_rwlv",
}

@dataclass
class RawSources:
    tray_scans: pd.DataFrame
    chip_inventory: pd.DataFrame
    chip_updates: pd.DataFrame
    bets: pd.DataFrame


@dataclass
class SourceBundle:
    sources: RawSources
    topology: pd.DataFrame
    profile: dict


class FeatureRepository:
    def expand_chip_ids(self, updates: pd.DataFrame) -> pd.DataFrame:
        if updates is None or updates.empty:
            return pd.DataFrame(columns=["table_id", "chip_id"])
        work = updates[updates["update_type"].astype(str).str.upper().eq("CHIPS_OUT")].copy()
        work["table_id"] = pd.to_numeric(work["from_table_id"], errors="coerce")
        work = work[work["table_id"].notna()].copy()
        if work.empty:
            return pd.DataFrame(columns=["table_id", "chip_id"])
        work["table_id"] = work["table_id"].astype(int)
        work["chip_id_list"] = work["chips_raw"].apply(normalize_chip_list)
        expanded = work[["table_id", "chip_id_list"]].explode("chip_id_list")
        expanded["chip_id"] = expanded["chip_id_list"].astype(str).str.strip()
        return expanded[expanded["chip_id"].ne("")][["table_id", "chip_id"]]


def parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def normalize_chip_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    raw = raw.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
    return [part.strip() for part in raw.split(",") if part.strip()]

def df_profile(df: pd.DataFrame, ts_col: str | None = None) -> dict:
    profile = {"rows": int(len(df))}
    if "table_id" in df.columns:
        profile["table_count"] = int(pd.to_numeric(df["table_id"], errors="coerce").nunique())
    if ts_col and ts_col in df.columns and not df.empty:
        ts = to_utc(df[ts_col])
        profile["min_ts"] = str(ts.min()) if ts.notna().any() else None
        profile["max_ts"] = str(ts.max()) if ts.notna().any() else None
    return profile

def normalize_sources(
    tray: pd.DataFrame,
    chip_inventory: pd.DataFrame,
    chip_updates: pd.DataFrame,
    bets: pd.DataFrame,
) -> RawSources:
    if not tray.empty:
        tray["table_id"] = pd.to_numeric(tray["table_id"], errors="coerce")
        tray["tray_balance"] = pd.to_numeric(tray["tray_balance"], errors="coerce")
        tray["tray_ts"] = to_utc(tray["tray_ts"])
        tray = tray[tray["table_id"].notna() & tray["tray_ts"].notna() & tray["tray_balance"].notna()].copy()
        tray["table_id"] = tray["table_id"].astype(int)

    if not chip_inventory.empty:
        chip_inventory["table_id"] = pd.to_numeric(chip_inventory["table_id"], errors="coerce")
        chip_inventory["denom"] = pd.to_numeric(chip_inventory["denom"], errors="coerce")
        chip_inventory["updated_ts"] = to_utc(chip_inventory["updated_ts"])
        chip_inventory = chip_inventory[chip_inventory["table_id"].notna() & chip_inventory["denom"].notna()].copy()
        chip_inventory["table_id"] = chip_inventory["table_id"].astype(int)

    if not chip_updates.empty:
        chip_updates["update_ts"] = to_utc(chip_updates["update_ts"])
        chip_updates["update_type"] = chip_updates["update_type"].astype(str).str.upper()
        chip_updates["update_value"] = pd.to_numeric(chip_updates["update_value"], errors="coerce").fillna(0.0)
        chip_updates["from_table_id"] = pd.to_numeric(chip_updates["from_table_id"], errors="coerce")
        chip_updates["to_table_id"] = pd.to_numeric(chip_updates["to_table_id"], errors="coerce")

    if not bets.empty:
        bets["table_id"] = pd.to_numeric(bets["table_id"], errors="coerce")
        bets["payout_ts"] = to_utc(bets["payout_ts"])
        bets["casino_win"] = pd.to_numeric(bets["casino_win"], errors="coerce").fillna(0.0)
        bets["casino_loss"] = pd.to_numeric(bets["casino_loss"], errors="coerce").fillna(0.0)
        bets = bets[bets["table_id"].notna() & bets["payout_ts"].notna()].copy()
        bets["table_id"] = bets["table_id"].astype(int)

    return RawSources(
        tray_scans=tray,
        chip_inventory=chip_inventory,
        chip_updates=chip_updates,
        bets=bets,
    )





