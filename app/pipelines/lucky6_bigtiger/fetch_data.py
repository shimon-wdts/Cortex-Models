from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd


SOURCE_TABLE = "t_game"

CANONICAL_ALIASES: dict[str, list[str]] = {
    "GameId": ["GameId", "game_id", "gameid", "game_uuid", "GameUuid"],
    "ShoeId": ["ShoeId", "shoe_id", "shoeid"],
    "ShoeGameCount": [
        "ShoeGameCount",
        "shoe_game_count",
        "shoe_game_cnt",
        "game_count",
        "game_no",
        "game_num",
        "game_number",
        "hand_id",
        "hand_no",
        "shoe_hand_no",
    ],
    "CardP1": ["CardP1", "card_p1", "cardp1", "player_card_1", "player_card1", "p_card_1"],
    "CardP2": ["CardP2", "card_p2", "cardp2", "player_card_2", "player_card2", "p_card_2"],
    "CardP3": ["CardP3", "card_p3", "cardp3", "player_card_3", "player_card3", "p_card_3"],
    "CardB1": ["CardB1", "card_b1", "cardb1", "banker_card_1", "banker_card1", "b_card_1"],
    "CardB2": ["CardB2", "card_b2", "cardb2", "banker_card_2", "banker_card2", "b_card_2"],
    "CardB3": ["CardB3", "card_b3", "cardb3", "banker_card_3", "banker_card3", "b_card_3"],
    "PlayerScore": ["PlayerScore", "player_score", "player_total", "player_points", "p_score"],
    "BankerScore": ["BankerScore", "banker_score", "banker_total", "banker_points", "b_score"],
    "Outcome": ["Outcome", "outcome", "game_result", "GameResult", "result", "winner", "winning_side"],
    "GameStartDtm": ["GameStartDtm", "game_start_dtm", "game_start_time", "created_at", "start_time"],
    "PayoutCompleteDtm": ["PayoutCompleteDtm", "payout_complete_dtm", "payout_ts", "completed_at"],
    "GamingDay": ["GamingDay", "gaming_day"],
    "TableId": ["TableId", "table_id"],
    "TableName": ["TableName", "table_name"],
    "PitName": ["PitName", "pit_name"],
    "GamingArea": ["GamingArea", "gaming_area"],
    "GameType": ["GameType", "game_type"],
    "GameStatus": ["GameStatus", "game_status"],
}

CANONICAL_COLUMNS = list(CANONICAL_ALIASES)


def first_present(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    lower_to_key = {str(key).lower(): key for key in row.keys()}
    for alias in aliases:
        key = lower_to_key.get(alias.lower())
        if key is not None:
            return row.get(key)
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value if value is not None else "").strip()
        return float(text) if text else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value if value is not None else "").strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def normalize_row(row: Mapping[str, Any], fallback_hand_id: int | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        value = first_present(row, aliases)
        if value is None and canonical == "ShoeGameCount" and fallback_hand_id is not None:
            value = fallback_hand_id
        out[canonical] = "" if value is None else str(value)

    if not out.get("Outcome"):
        player = safe_float(out.get("PlayerScore"))
        banker = safe_float(out.get("BankerScore"))
        if player > banker:
            out["Outcome"] = "PLAYER"
        elif banker > player:
            out["Outcome"] = "BANKER"
        elif player == banker and (out.get("PlayerScore") or out.get("BankerScore")):
            out["Outcome"] = "TIE"
    return out


def normalize_t_game_rows(rows: pd.DataFrame) -> list[dict[str, str]]:
    if rows is None or rows.empty:
        return []

    fallback_by_shoe: dict[str, int] = {}
    normalized: list[dict[str, str]] = []
    for raw in rows.to_dict(orient="records"):
        raw_shoe = str(first_present(raw, CANONICAL_ALIASES["ShoeId"]) or "")
        fallback_by_shoe[raw_shoe] = fallback_by_shoe.get(raw_shoe, 0) + 1
        normalized.append(normalize_row(raw, fallback_by_shoe[raw_shoe]))

    normalized.sort(
        key=lambda row: (
            str(row.get("ShoeId") or ""),
            safe_int(row.get("ShoeGameCount"), 10**9),
            str(row.get("GameStartDtm") or ""),
            str(row.get("GameId") or ""),
        )
    )
    return normalized


def to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)
