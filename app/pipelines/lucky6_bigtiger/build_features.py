#!/usr/bin/env python3
"""Live, row-by-row feature building for Lucky 6 production inference."""

from __future__ import annotations

import copy
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from app.pipelines.lucky6_bigtiger.fetch_data import to_utc_timestamp

INITIAL_DECKS = 8
CARDS_PER_DECK = 52
TOTAL_CARDS = INITIAL_DECKS * CARDS_PER_DECK
BASE_PER_RANK = 4 * INITIAL_DECKS
AVG_CARDS_PER_HAND = 5.5
CUT_CARDS = CARDS_PER_DECK
PLAYABLE_CARDS = TOTAL_CARDS - CUT_CARDS
EPS = 1e-6
SEQ_LEN = 30

RANKS_FULL = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
RANK_TO_INDEX = {rank: idx for idx, rank in enumerate(RANKS_FULL)}
RANK_VALUE = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 0,
    "J": 0,
    "Q": 0,
    "K": 0,
}

L6_TAGS_LVL1 = np.array([0, 0, 0, 0, 0, -3, +1, +1, +1, 0], dtype=np.float32)
L6_TAGS_LVL2 = np.array([-2, -1, -1, -4, -1, -1, -18, +13, +12, +9], dtype=np.float32)

LUCKY6_TOTAL_RETURN_2 = 13.0
LUCKY6_TOTAL_RETURN_3 = 24.0
BIG_TIGER_TOTAL_RETURN = 51.0
VALUE_TO_RANK_INDEX = np.array([9, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)
MAX_COMB_DRAWS = 6

SEQ_FEATURES = [
    "banker_win",
    "tie",
    "natural",
    "pt_final",
    "bt_final",
    "p_cards",
    "b_cards",
    "player_drew",
    "banker_drew",
    "cards_consumed_norm",
    "decks_remaining_norm",
    "cards_remaining_norm",
]

COMBINATORIC_FEATURES = [
    "comb_p_l6_2",
    "comb_p_l6_3",
    "comb_p_l6_any",
    "comb_p_big_tiger",
    "comb_ev_lucky6",
    "comb_ev_big_tiger",
]

ENGINEERED_FEATURES = [
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
    "comb_ev_lucky6_x_pct",
    "comb_ev_big_tiger_x_pct",
    "six_density_x_pct",
    "low_card_density_x_pct",
    "five_six_seven_density",
    "five_six_seven_density_x_pct",
    "rank_entropy_x_pct",
    "draw_volatility_proxy",
    "banker_draw_support",
    "player_draw_support",
    "draw_support_gap",
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
]

STATIC_GLOBAL_FEATURES = [
    "cards_remaining_norm",
    "decks_remaining_norm",
    "pct_shoe_done",
    "hands_remaining_norm",
    "six_density",
    "ten_density",
    "low_density",
    "high_density",
    "rank_entropy",
    "l6_rc_lvl1",
    "l6_tc_lvl1",
    "l6_rc_lvl2",
    "l6_tc_lvl2",
    "player_win_rate_5",
    "banker_win_rate_5",
    "tie_rate_5",
    "natural_rate_5",
    "avg_cards_consumed_5",
    "player_win_rate_10",
    "banker_win_rate_10",
    "tie_rate_10",
    "natural_rate_10",
    "avg_cards_consumed_10",
    "delta_player_win_rate_5v5",
    "delta_banker_win_rate_5v5",
    "delta_tie_rate_5v5",
    "delta_natural_rate_5v5",
    "delta_avg_cards_consumed_5v5",
    "six_x_decksrem",
    "six_x_pctdone",
    "tc1_x_pctdone",
    "tc2_x_pctdone",
] + COMBINATORIC_FEATURES + ENGINEERED_FEATURES


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _normalize_rank(card_text: str) -> Optional[str]:
    text = (card_text or "").strip()
    if not text:
        return None
    rank = text[:-1].upper() if len(text) >= 2 else text.upper()
    if rank == "T":
        rank = "10"
    return rank if rank in RANK_TO_INDEX else None


def _card_ranks_from_row(row: Dict[str, str], columns: Iterable[str]) -> List[str]:
    ranks: List[str] = []
    for column in columns:
        rank = _normalize_rank(row.get(column, ""))
        if rank is not None:
            ranks.append(rank)
    return ranks


def _hand_total(ranks: Iterable[str]) -> int:
    return sum(RANK_VALUE[rank] for rank in ranks) % 10


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / s
    return float(-(p * np.log(p)).sum())


def _mean_history(rows: List[Dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(row.get(key, 0.0) for row in rows) / len(rows))


def _banker_draws_with_player_third(bt: int, p3_value: int) -> bool:
    if bt <= 2:
        return True
    if bt == 3:
        return p3_value != 8
    if bt == 4:
        return 2 <= p3_value <= 7
    if bt == 5:
        return 4 <= p3_value <= 7
    if bt == 6:
        return p3_value in (6, 7)
    return False


def _add_value_pattern_term(terms: dict[tuple[int, ...], int], values: tuple[int, ...]) -> None:
    counts = [0] * len(RANKS)
    for value in values:
        counts[int(VALUE_TO_RANK_INDEX[value])] += 1
    terms[tuple(counts)] += 1


def _build_combinatoric_event_terms() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l6_2_terms: dict[tuple[int, ...], int] = defaultdict(int)
    l6_3_terms: dict[tuple[int, ...], int] = defaultdict(int)

    for p1, b1, p2, b2 in product(range(10), repeat=4):
        pt_initial = (p1 + p2) % 10
        bt_initial = (b1 + b2) % 10

        if pt_initial <= 5 and bt_initial == 6:
            for p3 in range(10):
                pt_final = (pt_initial + p3) % 10
                if p3 in (6, 7):
                    continue
                if pt_final < 6:
                    _add_value_pattern_term(l6_2_terms, (p1, b1, p2, b2, p3))

        if pt_initial <= 5 and bt_initial not in (8, 9):
            for p3 in range(10):
                pt_final = (pt_initial + p3) % 10
                if pt_final >= 6:
                    continue
                if not _banker_draws_with_player_third(bt_initial, p3):
                    continue
                b3 = (6 - bt_initial) % 10
                _add_value_pattern_term(l6_3_terms, (p1, b1, p2, b2, p3, b3))

    def pack(terms: dict[tuple[int, ...], int]) -> tuple[np.ndarray, np.ndarray]:
        items = sorted(terms.items())
        counts = np.asarray([k for k, _ in items], dtype=np.int8)
        multipliers = np.asarray([v for _, v in items], dtype=np.float64)
        return counts, multipliers

    l6_2_counts, l6_2_mult = pack(l6_2_terms)
    l6_3_counts, l6_3_mult = pack(l6_3_terms)
    return l6_2_counts, l6_2_mult, l6_3_counts, l6_3_mult


L6_2_TERM_COUNTS, L6_2_TERM_MULT, L6_3_TERM_COUNTS, L6_3_TERM_MULT = _build_combinatoric_event_terms()


def _pack_draw_terms(draw_terms: dict[int, dict[tuple[int, ...], int]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    packed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for draws, terms in sorted(draw_terms.items()):
        if not terms:
            continue
        items = sorted(terms.items())
        counts = np.asarray([k for k, _ in items], dtype=np.int8)
        multipliers = np.asarray([v for _, v in items], dtype=np.float64)
        packed[int(draws)] = (counts, multipliers)
    return packed


def _build_player6_event_terms() -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    p6_2_terms: dict[int, dict[tuple[int, ...], int]] = defaultdict(lambda: defaultdict(int))
    p6_3_terms: dict[int, dict[tuple[int, ...], int]] = defaultdict(lambda: defaultdict(int))

    for p1, b1, p2, b2 in product(range(10), repeat=4):
        pt_initial = (p1 + p2) % 10
        bt_initial = (b1 + b2) % 10
        if bt_initial in (8, 9):
            continue

        if pt_initial == 6 and bt_initial <= 5:
            for b3 in range(10):
                bt_final = (bt_initial + b3) % 10
                if bt_final < 6:
                    counts = [0] * len(RANKS)
                    for value in (p1, b1, p2, b2, b3):
                        counts[int(VALUE_TO_RANK_INDEX[value])] += 1
                    p6_2_terms[5][tuple(counts)] += 1

        if pt_initial <= 5:
            for p3 in range(10):
                pt_final = (pt_initial + p3) % 10
                if pt_final != 6:
                    continue
                if _banker_draws_with_player_third(bt_initial, p3):
                    for b3 in range(10):
                        bt_final = (bt_initial + b3) % 10
                        if bt_final < 6:
                            counts = [0] * len(RANKS)
                            for value in (p1, b1, p2, b2, p3, b3):
                                counts[int(VALUE_TO_RANK_INDEX[value])] += 1
                            p6_3_terms[6][tuple(counts)] += 1
                elif bt_initial < 6:
                    counts = [0] * len(RANKS)
                    for value in (p1, b1, p2, b2, p3):
                        counts[int(VALUE_TO_RANK_INDEX[value])] += 1
                    p6_3_terms[5][tuple(counts)] += 1

    return _pack_draw_terms(p6_2_terms), _pack_draw_terms(p6_3_terms)


PLAYER6_2_TERM_GROUPS, PLAYER6_3_TERM_GROUPS = _build_player6_event_terms()


def _comb_event_probability(
    fall_by_rank: np.ndarray,
    total_falling: np.ndarray,
    term_counts: np.ndarray,
    term_multipliers: np.ndarray,
    draws: int,
) -> float:
    denom = float(total_falling[draws])
    if denom <= 0.0:
        return 0.0
    products = np.ones(term_counts.shape[0], dtype=np.float64)
    for rank_idx in range(len(RANKS)):
        products *= fall_by_rank[rank_idx, term_counts[:, rank_idx]]
    return float(np.dot(term_multipliers, products) / denom)


def _comb_event_probability_groups(
    fall_by_rank: np.ndarray,
    total_falling: np.ndarray,
    groups: dict[int, tuple[np.ndarray, np.ndarray]],
) -> float:
    total = 0.0
    for draws, (term_counts, term_multipliers) in groups.items():
        total += _comb_event_probability(
            fall_by_rank,
            total_falling,
            term_counts,
            term_multipliers,
            draws=draws,
        )
    return float(total)


def _combinatoric_features(remain_vec: np.ndarray) -> np.ndarray:
    counts = np.rint(remain_vec).astype(np.int16, copy=False)
    total_cards = int(np.sum(counts))

    fall_by_rank = np.ones((len(RANKS), MAX_COMB_DRAWS + 1), dtype=np.float64)
    for rank_idx, count in enumerate(counts):
        acc = 1.0
        for draws in range(1, MAX_COMB_DRAWS + 1):
            if count < draws:
                acc = 0.0
            else:
                acc *= float(count - draws + 1)
            fall_by_rank[rank_idx, draws] = acc

    total_falling = np.ones(MAX_COMB_DRAWS + 1, dtype=np.float64)
    acc_total = 1.0
    for draws in range(1, MAX_COMB_DRAWS + 1):
        if total_cards < draws:
            acc_total = 0.0
        else:
            acc_total *= float(total_cards - draws + 1)
        total_falling[draws] = acc_total

    p_l6_2 = _comb_event_probability(fall_by_rank, total_falling, L6_2_TERM_COUNTS, L6_2_TERM_MULT, draws=5)
    p_l6_3 = _comb_event_probability(fall_by_rank, total_falling, L6_3_TERM_COUNTS, L6_3_TERM_MULT, draws=6)
    p_l6_any = p_l6_2 + p_l6_3
    ev_lucky6 = -1.0 + (LUCKY6_TOTAL_RETURN_2 * p_l6_2) + (LUCKY6_TOTAL_RETURN_3 * p_l6_3)
    ev_big_tiger = -1.0 + (BIG_TIGER_TOTAL_RETURN * p_l6_3)
    player_p6_2 = _comb_event_probability_groups(fall_by_rank, total_falling, PLAYER6_2_TERM_GROUPS)
    player_p6_3 = _comb_event_probability_groups(fall_by_rank, total_falling, PLAYER6_3_TERM_GROUPS)
    player_p6_any = player_p6_2 + player_p6_3
    player_ev6_proxy = -1.0 + (LUCKY6_TOTAL_RETURN_2 * player_p6_2) + (LUCKY6_TOTAL_RETURN_3 * player_p6_3)

    return np.array(
        [p_l6_2, p_l6_3, p_l6_any, p_l6_3, ev_lucky6, ev_big_tiger, player_p6_2, player_p6_3, player_p6_any, player_ev6_proxy],
        dtype=np.float32,
    )


def _ev_history_features(
    current_l6: float,
    current_bt: float,
    ev_l6_history: List[float],
    ev_bt_history: List[float],
) -> List[float]:
    def delta(history: List[float], current: float, lag: int) -> float:
        if len(history) < lag:
            return 0.0
        return float(current - history[-lag])

    def roll_mean(history: List[float], current: float, window: int) -> float:
        vals = history[-(window - 1) :] + [float(current)]
        return float(np.mean(vals)) if vals else float(current)

    return [
        delta(ev_l6_history, current_l6, 1),
        delta(ev_l6_history, current_l6, 3),
        delta(ev_l6_history, current_l6, 5),
        delta(ev_l6_history, current_l6, 10),
        delta(ev_bt_history, current_bt, 1),
        delta(ev_bt_history, current_bt, 3),
        delta(ev_bt_history, current_bt, 5),
        delta(ev_bt_history, current_bt, 10),
        roll_mean(ev_l6_history, current_l6, 3),
        roll_mean(ev_l6_history, current_l6, 5),
        roll_mean(ev_l6_history, current_l6, 10),
        roll_mean(ev_bt_history, current_bt, 3),
        roll_mean(ev_bt_history, current_bt, 5),
        roll_mean(ev_bt_history, current_bt, 10),
    ]


@dataclass
class ShoeState:
    shoe_id: Optional[str] = None
    cards_seen_total: int = 0
    completed_hands: int = 0
    remain_full: np.ndarray = field(default_factory=lambda: np.full(13, BASE_PER_RANK, dtype=np.float32))
    history: Deque[Dict[str, float]] = field(default_factory=lambda: deque(maxlen=SEQ_LEN))
    ev_l6_history: List[float] = field(default_factory=list)
    ev_bt_history: List[float] = field(default_factory=list)

    def reset(self, shoe_id: str) -> None:
        self.shoe_id = shoe_id
        self.cards_seen_total = 0
        self.completed_hands = 0
        self.remain_full = np.full(13, BASE_PER_RANK, dtype=np.float32)
        self.history = deque(maxlen=SEQ_LEN)
        self.ev_l6_history = []
        self.ev_bt_history = []


class LiveFeatureBuilder:
    """Incrementally rebuilds the same logical inputs used in training."""

    def __init__(self, seq_len: int = SEQ_LEN) -> None:
        self.seq_len = seq_len
        self.states: Dict[str, ShoeState] = {}
        self.state = ShoeState()

    def process_completed_row(self, row: Dict[str, str]) -> Dict[str, object]:
        shoe_id = str(row.get("ShoeId", "")).strip()
        if not shoe_id:
            raise ValueError("Row is missing ShoeId")

        shoe_reset = shoe_id not in self.states
        if shoe_reset:
            state = ShoeState()
            state.reset(shoe_id)
            self.states[shoe_id] = state
        self.state = self.states[shoe_id]

        hand_id = _safe_int(row.get("ShoeGameCount"), self.state.completed_hands + 1)
        next_hand_id = hand_id + 1
        player_ranks = _card_ranks_from_row(row, ("CardP1", "CardP2", "CardP3"))
        banker_ranks = _card_ranks_from_row(row, ("CardB1", "CardB2", "CardB3"))
        all_ranks = player_ranks + banker_ranks
        if not all_ranks:
            return {
                "shoe_id": shoe_id,
                "shoe_reset": shoe_reset,
                "hand_id": hand_id,
                "next_hand_id": next_hand_id,
                "cards_remaining": float(max(0, TOTAL_CARDS - self.state.cards_seen_total)),
                "decks_remaining": float(max(0, TOTAL_CARDS - self.state.cards_seen_total) / CARDS_PER_DECK),
                "history_size": len(self.state.history),
                "skip_reason": "no_usable_cards",
            }

        for rank in all_ranks:
            idx = RANK_TO_INDEX[rank]
            if self.state.remain_full[idx] <= 0:
                raise ValueError(f"Card inventory went negative for rank {rank} in shoe {shoe_id}")
            self.state.remain_full[idx] -= 1

        cards_consumed = len(all_ranks)
        self.state.cards_seen_total += cards_consumed
        self.state.completed_hands += 1

        pt_final = _safe_int(row.get("PlayerScore"), _hand_total(player_ranks))
        bt_final = _safe_int(row.get("BankerScore"), _hand_total(banker_ranks))

        outcome = str(row.get("Outcome", "")).strip().upper()
        banker_win = 1.0 if outcome == "BANKER" else 0.0
        tie = 1.0 if outcome == "TIE" else 0.0
        player_win = 1.0 if outcome == "PLAYER" else 0.0

        natural = 0.0
        if len(player_ranks) >= 2 and len(banker_ranks) >= 2:
            natural = 1.0 if (_hand_total(player_ranks[:2]) in (8, 9) or _hand_total(banker_ranks[:2]) in (8, 9)) else 0.0

        cards_remaining = float(max(0, TOTAL_CARDS - self.state.cards_seen_total))
        decks_remaining = float(cards_remaining / CARDS_PER_DECK)

        hist_row = {
            "player_win": player_win,
            "banker_win": banker_win,
            "tie": tie,
            "natural": natural,
            "pt_final": float(pt_final),
            "bt_final": float(bt_final),
            "p_cards": float(len(player_ranks)),
            "b_cards": float(len(banker_ranks)),
            "player_drew": 1.0 if len(player_ranks) == 3 else 0.0,
            "banker_drew": 1.0 if len(banker_ranks) == 3 else 0.0,
            "cards_consumed_norm": float(cards_consumed / 6.0),
            "decks_remaining_norm": float(decks_remaining / INITIAL_DECKS),
            "cards_remaining_norm": float(cards_remaining / TOTAL_CARDS),
        }
        self.state.history.append(hist_row)

        x_static = self._build_static_matrix(cards_remaining, decks_remaining)
        x_seq = self._build_sequence_matrix()

        return {
            "shoe_id": shoe_id,
            "shoe_reset": shoe_reset,
            "hand_id": hand_id,
            "next_hand_id": next_hand_id,
            "cards_remaining": cards_remaining,
            "decks_remaining": decks_remaining,
            "history_size": len(self.state.history),
            "X_static": x_static,
            "X_seq": x_seq,
            "skip_reason": None,
        }

    def _build_sequence_matrix(self) -> np.ndarray:
        x_seq = np.zeros((self.seq_len, len(SEQ_FEATURES)), dtype=np.float32)
        hist = list(self.state.history)
        if not hist:
            return x_seq

        tail = hist[-self.seq_len :]
        start = self.seq_len - len(tail)
        for row_idx, hist_row in enumerate(tail, start=start):
            x_seq[row_idx] = np.array([hist_row[name] for name in SEQ_FEATURES], dtype=np.float32)
        return x_seq

    def _build_static_matrix(self, cards_remaining: float, decks_remaining: float) -> np.ndarray:
        remain_vec = np.array(
            [
                float(self.state.remain_full[0]),
                float(self.state.remain_full[1]),
                float(self.state.remain_full[2]),
                float(self.state.remain_full[3]),
                float(self.state.remain_full[4]),
                float(self.state.remain_full[5]),
                float(self.state.remain_full[6]),
                float(self.state.remain_full[7]),
                float(self.state.remain_full[8]),
                float(self.state.remain_full[9] + self.state.remain_full[10] + self.state.remain_full[11] + self.state.remain_full[12]),
            ],
            dtype=np.float32,
        )

        decks_remaining_norm = float(decks_remaining / INITIAL_DECKS)
        decks_remaining_safe = max(decks_remaining, EPS)
        cards_remaining_norm = float(cards_remaining / TOTAL_CARDS)

        frac_vec = remain_vec / (cards_remaining + EPS)
        expected_frac = BASE_PER_RANK / TOTAL_CARDS

        six_density = float(frac_vec[5])
        ten_density = float(frac_vec[9])
        low_density = float(frac_vec[0:5].sum())
        high_density = float(frac_vec[6:10].sum())
        five_six_seven_density = float(frac_vec[4:7].sum())
        rank_entropy = _entropy(frac_vec.astype(np.float64))

        seen_vec = (BASE_PER_RANK - remain_vec).astype(np.float32)
        l6_rc_lvl1 = float(np.dot(L6_TAGS_LVL1, seen_vec))
        l6_tc_lvl1 = float(l6_rc_lvl1 / decks_remaining_safe)
        l6_rc_lvl2 = float(np.dot(L6_TAGS_LVL2, seen_vec))
        l6_tc_lvl2 = float(l6_rc_lvl2 / decks_remaining_safe)

        hist = list(self.state.history)
        last5 = hist[-5:] if len(hist) >= 5 else hist
        last10 = hist[-10:] if len(hist) >= 10 else hist
        prev5 = hist[-10:-5] if len(hist) >= 10 else last5

        player_win_rate_5 = _mean_history(last5, "player_win")
        banker_win_rate_5 = _mean_history(last5, "banker_win")
        tie_rate_5 = _mean_history(last5, "tie")
        natural_rate_5 = _mean_history(last5, "natural")
        avg_cards_consumed_5 = _mean_history(last5, "cards_consumed_norm")

        player_win_rate_10 = _mean_history(last10, "player_win")
        banker_win_rate_10 = _mean_history(last10, "banker_win")
        tie_rate_10 = _mean_history(last10, "tie")
        natural_rate_10 = _mean_history(last10, "natural")
        avg_cards_consumed_10 = _mean_history(last10, "cards_consumed_norm")

        delta_player_win_rate_5v5 = player_win_rate_5 - _mean_history(prev5, "player_win")
        delta_banker_win_rate_5v5 = banker_win_rate_5 - _mean_history(prev5, "banker_win")
        delta_tie_rate_5v5 = tie_rate_5 - _mean_history(prev5, "tie")
        delta_natural_rate_5v5 = natural_rate_5 - _mean_history(prev5, "natural")
        delta_avg_cards_consumed_5v5 = avg_cards_consumed_5 - _mean_history(prev5, "cards_consumed_norm")

        cards_removed = max(0.0, TOTAL_CARDS - cards_remaining)
        cards_to_cut = max(0.0, cards_remaining - CUT_CARDS)
        playable_cards_safe = max(1.0, float(PLAYABLE_CARDS))
        pct_shoe_done = float(min(1.0, cards_removed / playable_cards_safe))
        hands_remaining_norm = float(min(1.0, cards_to_cut / playable_cards_safe))

        six_x_decksrem = six_density * decks_remaining_norm
        six_x_pctdone = six_density * pct_shoe_done
        tc1_x_pctdone = l6_tc_lvl1 * pct_shoe_done
        tc2_x_pctdone = l6_tc_lvl2 * pct_shoe_done
        comb_vec = _combinatoric_features(remain_vec)

        p_l6_2 = float(comb_vec[0])
        p_l6_3 = float(comb_vec[1])
        p_l6_any = float(comb_vec[2])
        ev_lucky6 = float(comb_vec[4])
        ev_big_tiger = float(comb_vec[5])
        player_p6_2 = float(comb_vec[6])
        player_p6_3 = float(comb_vec[7])
        player_p6_any = float(comb_vec[8])
        player_ev6_proxy = float(comb_vec[9])

        banker_player_p6_2_gap = p_l6_2 - player_p6_2
        banker_player_p6_3_gap = p_l6_3 - player_p6_3
        banker_player_p6_any_gap = p_l6_any - player_p6_any
        total_p6_2_richness = p_l6_2 + player_p6_2
        total_p6_3_richness = p_l6_3 + player_p6_3
        total_p6_any_richness = p_l6_any + player_p6_any
        comb_ev_lucky6_margin = ev_lucky6
        comb_ev_big_tiger_margin = ev_big_tiger
        comb_ev_lucky6_x_pct = ev_lucky6 * pct_shoe_done
        comb_ev_big_tiger_x_pct = ev_big_tiger * pct_shoe_done
        six_density_x_pct = six_density * pct_shoe_done
        low_card_density_x_pct = low_density * pct_shoe_done
        five_six_seven_density_x_pct = five_six_seven_density * pct_shoe_done
        rank_entropy_x_pct = rank_entropy * pct_shoe_done
        banker_draw_support = p_l6_3 / max(p_l6_any, EPS)
        player_draw_support = player_p6_3 / max(player_p6_any, EPS)
        draw_support_gap = banker_draw_support - player_draw_support
        draw_volatility_proxy = total_p6_3_richness / max(total_p6_any_richness, EPS)
        ev_history_vec = _ev_history_features(ev_lucky6, ev_big_tiger, self.state.ev_l6_history, self.state.ev_bt_history)
        phase_early = 1.0 if pct_shoe_done < 0.33 else 0.0
        phase_mid = 1.0 if 0.33 <= pct_shoe_done < 0.66 else 0.0
        phase_late = 1.0 if pct_shoe_done >= 0.66 else 0.0

        globals_vec = np.array(
            [
                cards_remaining_norm,
                decks_remaining_norm,
                pct_shoe_done,
                hands_remaining_norm,
                six_density,
                ten_density,
                low_density,
                high_density,
                rank_entropy,
                l6_rc_lvl1,
                l6_tc_lvl1,
                l6_rc_lvl2,
                l6_tc_lvl2,
                player_win_rate_5,
                banker_win_rate_5,
                tie_rate_5,
                natural_rate_5,
                avg_cards_consumed_5,
                player_win_rate_10,
                banker_win_rate_10,
                tie_rate_10,
                natural_rate_10,
                avg_cards_consumed_10,
                delta_player_win_rate_5v5,
                delta_banker_win_rate_5v5,
                delta_tie_rate_5v5,
                delta_natural_rate_5v5,
                delta_avg_cards_consumed_5v5,
                six_x_decksrem,
                six_x_pctdone,
                tc1_x_pctdone,
                tc2_x_pctdone,
                *comb_vec[: len(COMBINATORIC_FEATURES)],
                player_p6_2,
                player_p6_3,
                player_p6_any,
                player_ev6_proxy,
                banker_player_p6_2_gap,
                banker_player_p6_3_gap,
                banker_player_p6_any_gap,
                total_p6_2_richness,
                total_p6_3_richness,
                total_p6_any_richness,
                comb_ev_lucky6_margin,
                comb_ev_big_tiger_margin,
                comb_ev_lucky6_x_pct,
                comb_ev_big_tiger_x_pct,
                six_density_x_pct,
                low_card_density_x_pct,
                five_six_seven_density,
                five_six_seven_density_x_pct,
                rank_entropy_x_pct,
                draw_volatility_proxy,
                banker_draw_support,
                player_draw_support,
                draw_support_gap,
                *ev_history_vec,
                phase_early,
                phase_mid,
                phase_late,
            ],
            dtype=np.float32,
        )
        self.state.ev_l6_history.append(ev_lucky6)
        self.state.ev_bt_history.append(ev_big_tiger)

        d_static = 5 + len(globals_vec)
        xs = np.zeros((len(RANKS), d_static), dtype=np.float32)
        std_est = math.sqrt(expected_frac * (1.0 - expected_frac) / (cards_remaining + EPS))
        for i in range(len(RANKS)):
            frac = float(frac_vec[i])
            delta = float(frac - expected_frac)
            z_like = float(delta / (std_est + EPS))
            xs[i, 0] = float(remain_vec[i] / BASE_PER_RANK)
            xs[i, 1] = frac
            xs[i, 2] = delta
            xs[i, 3] = z_like
            xs[i, 4] = float(i / (len(RANKS) - 1))
            xs[i, 5:] = globals_vec
        return xs


@dataclass
class Lucky6FeatureRecord:
    hand_id: int | str
    shoe_id: str
    game_id: str
    event_ts: pd.Timestamp | None
    game_start_ts: pd.Timestamp | None
    gaming_day: str
    table_id: str
    table_name: str
    pit_name: str
    gaming_area: str
    feature_row: np.ndarray | None
    skip_reason: str | None = None
    cards_remaining: float | None = None
    decks_remaining: float | None = None
    history_size: int | None = None
    source_row: dict[str, str] = field(default_factory=dict)


@dataclass
class Lucky6FeatureResult:
    records: list[Lucky6FeatureRecord]
    replayed_rows: int
    skipped_rows: int
    publish_start_ts: pd.Timestamp
    publish_end_ts: pd.Timestamp

    def __len__(self) -> int:
        return len(self.records)


def full_feature_row(built: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(built["X_static"], dtype=np.float32).reshape(1, -1),
            np.asarray(built["X_seq"], dtype=np.float32).reshape(1, -1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def build_feature_dataset(
    rows: list[dict[str, str]],
    *,
    publish_start_ts: pd.Timestamp,
    publish_end_ts: pd.Timestamp,
    hand_id_mode: str = "next",
) -> Lucky6FeatureResult:
    if hand_id_mode not in {"next", "current"}:
        raise ValueError("hand_id_mode must be 'next' or 'current'")

    start_ts = pd.Timestamp(publish_start_ts)
    end_ts = pd.Timestamp(publish_end_ts)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    builder = LiveFeatureBuilder()
    records: list[Lucky6FeatureRecord] = []
    skipped_rows = 0
    pending_by_shoe: dict[str, tuple[dict[str, str], pd.Timestamp, dict[str, Any]]] = {}

    for row in rows:
        event_ts = to_utc_timestamp(row.get("PayoutCompleteDtm"))
        game_start_ts = to_utc_timestamp(row.get("GameStartDtm"))
        shoe_key = str(row.get("ShoeId", "") or "")

        pending = pending_by_shoe.get(shoe_key)
        if pending is not None and game_start_ts is not None:
            source_row, source_event_ts, built = pending
            ready_ts = max(source_event_ts, game_start_ts)
            if start_ts <= ready_ts <= end_ts and _is_next_game(source_row, row, built):
                skip_reason = built.get("skip_reason")
                records.append(
                    _record_for_target_game(
                        row,
                        source_event_ts,
                        game_start_ts,
                        hand_id_mode,
                        None if skip_reason else full_feature_row(built),
                        str(skip_reason) if skip_reason else None,
                        built,
                    )
                )

        # An unfinished game supplies the target game ID for the previous hand's
        # prediction, but must not be applied to the card-history state itself.
        if event_ts is None or event_ts > end_ts:
            continue

        shoe_backup = copy.deepcopy(builder.states.get(shoe_key)) if shoe_key in builder.states else None
        built: dict[str, Any]
        try:
            built = builder.process_completed_row(row)
        except Exception as exc:
            skipped_rows += 1
            if shoe_backup is None:
                builder.states.pop(shoe_key, None)
                builder.state = ShoeState()
            else:
                builder.states[shoe_key] = shoe_backup
                builder.state = shoe_backup
            pending_by_shoe[shoe_key] = (row, event_ts, {"skip_reason": f"feature_error: {exc}"})
            continue

        skip_reason = built.get("skip_reason")
        if skip_reason:
            skipped_rows += 1
        pending_by_shoe[shoe_key] = (row, event_ts, built)

    return Lucky6FeatureResult(
        records=records,
        replayed_rows=len(rows),
        skipped_rows=skipped_rows,
        publish_start_ts=start_ts,
        publish_end_ts=end_ts,
    )


def _is_next_game(
    source_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    built: Mapping[str, Any],
) -> bool:
    expected_hand_id = _safe_optional_int(built.get("next_hand_id"))
    target_hand_id = _safe_optional_int(target_row.get("ShoeGameCount"))
    return (
        bool(str(target_row.get("GameId", "")).strip())
        and str(source_row.get("ShoeId", "")) == str(target_row.get("ShoeId", ""))
        and expected_hand_id is not None
        and target_hand_id == expected_hand_id
    )


def _record_for_target_game(
    target_row: dict[str, str],
    event_ts: pd.Timestamp | None,
    game_start_ts: pd.Timestamp | None,
    hand_id_mode: str,
    feature_row: np.ndarray | None,
    skip_reason: str | None,
    built: Mapping[str, Any] | None = None,
) -> Lucky6FeatureRecord:
    built = built or {}
    hand_key = "next_hand_id" if hand_id_mode == "next" else "hand_id"
    hand_id = built.get(hand_key, target_row.get("ShoeGameCount", ""))
    return Lucky6FeatureRecord(
        hand_id=hand_id,
        shoe_id=str(built.get("shoe_id", target_row.get("ShoeId", ""))),
        game_id=str(target_row.get("GameId", "")),
        event_ts=event_ts,
        game_start_ts=game_start_ts,
        gaming_day=str(target_row.get("GamingDay", "")),
        table_id=str(target_row.get("TableId", "")),
        table_name=str(target_row.get("TableName", "")),
        pit_name=str(target_row.get("PitName", "")),
        gaming_area=str(target_row.get("GamingArea", "")),
        feature_row=feature_row,
        skip_reason=skip_reason,
        cards_remaining=_safe_optional_float(built.get("cards_remaining")),
        decks_remaining=_safe_optional_float(built.get("decks_remaining")),
        history_size=_safe_optional_int(built.get("history_size")),
        source_row=target_row,
    )


def _safe_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None
