#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR
from .utils import apply_zscore, read_json, safe_write_csv


def infer_cohorts(feature_path: Path, model_path: Path, output_dir: Path) -> dict[str, object]:
    data = pd.read_csv(feature_path, low_memory=False)
    model = read_json(model_path)
    features = model["features"]
    x_df = apply_zscore(data, features, model["scalers"])
    x = x_df.to_numpy(dtype=float)
    centroids = np.asarray(model["centroids"], dtype=float)
    dist = np.sqrt(((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    labels = dist.argmin(axis=1)
    sorted_dist = np.sort(dist, axis=1)
    sorted_idx = np.argsort(dist, axis=1)
    data["cohort_id"] = labels
    data["cohort_distance"] = dist[np.arange(len(data)), labels]
    data["nearest_other_cohort_id"] = np.where(sorted_idx.shape[1] > 1, sorted_idx[:, 1], np.nan)
    data["nearest_other_cohort_distance"] = np.where(
        sorted_dist.shape[1] > 1, sorted_dist[:, 1], np.nan
    )
    data["cohort_margin_to_next"] = np.where(sorted_dist.shape[1] > 1, sorted_dist[:, 1] - sorted_dist[:, 0], np.nan)
    data["cohort_confidence_score"] = 1.0 / (1.0 + data["cohort_distance"])
    labels_map = {int(k): v for k, v in model.get("cohort_labels", {}).items()}
    data["cohort_model_label"] = data["cohort_id"].map(labels_map).fillna("Unknown")
    score_map = {int(k): float(v) for k, v in model.get("cohort_development_scores", {}).items()}
    rank_map = {int(k): int(v) for k, v in model.get("cohort_development_ranks", {}).items()}
    data["cohort_development_score"] = data["cohort_id"].map(score_map).fillna(0.0)
    data["cohort_development_rank"] = data["cohort_id"].map(rank_map).fillna(0).astype(int)

    target_ids = []
    target_distances = []
    target_margins = []
    target_scores = []
    target_ranks = []
    for row_idx, current in enumerate(labels):
        current_score = score_map.get(int(current), 0.0)
        better = [cid for cid, score in score_map.items() if score > current_score]
        if better:
            target = min(better, key=lambda cid: dist[row_idx, cid])
            target_ids.append(target)
            target_distances.append(float(dist[row_idx, target]))
            target_margins.append(float(dist[row_idx, target] - dist[row_idx, current]))
            target_scores.append(score_map.get(target, 0.0))
            target_ranks.append(rank_map.get(target, 0))
        else:
            target_ids.append(np.nan)
            target_distances.append(np.nan)
            target_margins.append(np.nan)
            target_scores.append(np.nan)
            target_ranks.append(np.nan)
    data["target_better_cohort_id"] = target_ids
    data["target_better_cohort_distance"] = target_distances
    data["target_better_cohort_margin"] = target_margins
    data["target_better_cohort_development_score"] = target_scores
    data["target_better_cohort_development_rank"] = target_ranks
    data["target_better_cohort_label"] = data["target_better_cohort_id"].map(labels_map).fillna("")
    edge_threshold = data["target_better_cohort_margin"].quantile(0.25)
    data["cohort_edge_to_better_flag"] = (
        data["target_better_cohort_id"].notna()
        & data["target_better_cohort_margin"].le(edge_threshold)
    )
    data["cohort_edge_score"] = 1.0 / (1.0 + data["target_better_cohort_margin"].clip(lower=0))
    out = output_dir / "player_cohorts.csv"
    safe_write_csv(data, out)
    return {"rows": int(len(data)), "output": str(out)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign behavioral cohorts to player-period features.")
    parser.add_argument("--features", type=Path, default=DEFAULT_OUTPUT_DIR / "player_period_features.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR / "cohort_model.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(infer_cohorts(args.features, args.model, args.output_dir))


if __name__ == "__main__":
    main()
