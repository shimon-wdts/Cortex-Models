from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipelines.cohorts.behavior_backtest import run_behavior_backtest  # noqa: E402


DEFAULT_COLUMNS = [
    "BetId",
    "SessionId",
    "PlayerId",
    "TableId",
    "GamingDay",
    "Wager",
    "TheoWin",
    "CasinoWin",
    "BetType",
    "TypeOfBet",
    "PayoutCompleteDtm",
    "SessionStartDtm",
    "SessionEndDtm",
    "GameType_game",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest historical recommendation-like behavior changes against AWT and ADT growth."
    )
    parser.add_argument("input", type=Path, help="Crowne master_sample_sessions.parquet path")
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/crowne_behavior_backtest"))
    parser.add_argument("--month", action="append", dest="months", help="YYYY-MM month; repeat for multiple months")
    parser.add_argument("--min-baseline-bets", type=int, default=30)
    args = parser.parse_args()

    try:
        data = pd.read_parquet(args.input, columns=DEFAULT_COLUMNS)
    except ImportError as exc:
        raise SystemExit("Reading the Crowne parquet requires the optional pyarrow package.") from exc

    summary, outcomes, metadata = run_behavior_backtest(
        data,
        months=args.months,
        min_baseline_bets=args.min_baseline_bets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "behavior_growth_summary.csv", index=False)
    outcomes.to_csv(args.output_dir / "player_behavior_outcomes.csv", index=False)
    (args.output_dir / "backtest_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote backtest outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
