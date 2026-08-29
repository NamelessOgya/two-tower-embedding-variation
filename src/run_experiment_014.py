"""
Plan 014 Experiment Runner: Stateless High-Precision & High-Diversity Evaluation
---------------------------------------------------------------------------------
Evaluates stateless methods (Semantic Stratified Partition and Multi-Head Stratified Partition Hybrid)
against Base, soft_jaccard, and Random Item Partition.

Metrics evaluated:
- recall_cum: 10-trial Cumulative Recall
- recall_avg: Per-trial Average Recall
- total_slate_precision: Total Slate Precision across 100 recommendation slots (%)
- gross_hits: Average positive hits per user across 100 recommendation slots
- ils: Intra-List Similarity
- hgts: Hit Ground-Truth Spread
- hgc: Hit Genre Coverage
- temporal_overlap & diversity

Usage:
    PYTHONPATH=. python3 src/run_experiment_014.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.run_experiment import load_data, get_train_pos, get_test_gt, SEEDS, K, N_TRIALS
from src.evaluate.metrics import (
    recall_at_k, recall_at_k_single, hit_at_k, temporal_overlap_rate,
)
from src.model.models_007 import TwoTowerModel
from src.model.models_008 import TwoTowerDivLoss
from src.model.models_012 import TwoTowerItemPartition
from src.model.models_014 import (
    TwoTowerSemanticStratifiedPartition,
    TwoTowerMultiHeadStratifiedPartition,
)
from src.run_experiment_013 import (
    parse_item_genres,
    compute_intra_list_similarity,
    compute_hit_gt_spread,
    compute_hit_genre_coverage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

N_SEEDS  = len(SEEDS)
OUT_DIR  = Path("report/plan_014")
DATA_DIR = Path("data/processed/movielens")


def run_experiment_014(model, test_gt, train_pos, user_embs, item_embs, item2genres, iid2idx, device):
    model.prepare(train_pos, user_embs, item_embs, device=device)
    index = model.build_index()

    norm_item_embs = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-9)

    seed_results = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        per_user = defaultdict(list)
        for user_idx, gt in test_gt.items():
            trial_lists = []
            trial_sets  = []
            all_hit_items = []
            for trial in range(N_TRIALS):
                recs = model.recommend(user_idx, trial, rng, index, K)
                trial_lists.append(recs)
                t_set = set(recs)
                trial_sets.append(t_set)
                hits = [it for it in recs if it in gt]
                all_hit_items.extend(hits)

            rc = recall_at_k(trial_sets, gt)
            ra = float(np.mean([recall_at_k_single(s, gt) for s in trial_sets]))
            gt_len = len(gt)

            # Slate Metrics
            slate_prec = (ra * gt_len / float(K)) * 100.0  # % of total slots containing GT items
            gross_hits = ra * gt_len * float(N_TRIALS)     # total hit count in 100 slots

            per_user["recall_cum"].append(rc)
            per_user["recall_avg"].append(ra)
            per_user["slate_precision"].append(slate_prec)
            per_user["gross_hits"].append(gross_hits)
            per_user["hit"].append(float(np.mean([hit_at_k(s, gt) for s in trial_sets])))
            per_user["temporal_overlap"].append(temporal_overlap_rate(trial_sets, K))
            per_user["diversity"].append(1.0 - temporal_overlap_rate(trial_sets, K))
            per_user["ils"].append(compute_intra_list_similarity(trial_lists, norm_item_embs))
            per_user["hgts"].append(compute_hit_gt_spread(all_hit_items, norm_item_embs))
            per_user["hgc"].append(compute_hit_genre_coverage(all_hit_items, item2genres, iid2idx))

        mean_r = {k2: float(np.mean(v)) for k2, v in per_user.items()}
        log.info(
            f"  seed={seed}  rc={mean_r['recall_cum']:.4f}  "
            f"ra={mean_r['recall_avg']:.4f}  "
            f"SlatePrec={mean_r['slate_precision']:.2f}%  "
            f"GrossHits={mean_r['gross_hits']:.2f}  "
            f"HGTS={mean_r['hgts']:.4f}  "
            f"HGC={mean_r['hgc']:.2f}"
        )
        seed_results.append(mean_r)

    keys = seed_results[0].keys()
    mean = {k2: float(np.mean([r[k2] for r in seed_results])) for k2 in keys}
    std  = {k2: float(np.std( [r[k2] for r in seed_results])) for k2 in keys}
    return {"mean": mean, "std": std}


def plot_plan_014_tradeoff(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0f1117")
    fig.suptitle("Plan 014: Stateless High-Precision & High-Diversity Recommendation Evaluation",
                 color="#ffffff", fontsize=16, fontweight="bold", y=0.98)

    models_order = [
        "TwoTower_d2_h64",
        "TT_divloss_soft_jaccard_l0p1_s0p05",
        "TT_item_partition_n10",
        "TT_semantic_stratified_partition_n10",
        "TT_multihead_stratified_partition_n10",
    ]
    labels = [
        "Base (no-div)",
        "soft_jaccard",
        "Random Partition",
        "Semantic Stratified",
        "MultiHead Stratified",
    ]
    colors = ["#555555", "#f1c40f", "#3498db", "#2ecc71", "#9b59b6"]

    metrics_config = [
        ("recall_cum", "Cumulative Recall@10 (rc) ↑", "1. Cumulative Recall (10-trial Coverage)"),
        ("recall_avg", "Per-trial Recall@10 (ra) ↑", "2. Per-Trial Average Recall"),
        ("slate_precision", "Total Slate Precision (%) ↑", "3. Total Slate Precision (out of 100 slots)"),
        ("hgts", "Hit GT Spread (HGTS) ↑", "4. Hit Ground-Truth Spread (HGTS)"),
    ]

    for ax_idx, (metric_key, ylabel, title) in enumerate(metrics_config):
        ax = axes[ax_idx // 2, ax_idx % 2]
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)

        vals = [all_results[m]["mean"][metric_key] for m in models_order if m in all_results]
        bars = ax.bar(labels[:len(vals)], vals, color=colors[:len(vals)], alpha=0.85, edgecolor="white", linewidth=0.8)

        ax.set_ylabel(ylabel, color="#e0e0e0", fontsize=10)
        ax.set_title(title, color="#ffffff", fontsize=11, fontweight="bold", pad=8)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", color="#e0e0e0")

        for bar in bars:
            h = bar.get_height()
            text_str = f"{h:.2f}%" if metric_key == "slate_precision" else f"{h:.4f}"
            ax.annotate(text_str,
                        (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9, color="#ffffff", fontweight="bold",
                        xytext=(0, 3), textcoords="offset points")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = out_dir / "tradeoff_plan_014.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Saved plot -> {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, user_embs, item_embs, uid2idx, iid2idx = load_data(DATA_DIR)
    train_pos = get_train_pos(df, uid2idx, iid2idx)
    test_gt   = get_test_gt(df, uid2idx, iid2idx)
    item2genres = parse_item_genres(DATA_DIR / "item_texts.parquet")

    # 1. Base Model
    log.info("Loading / Re-training Base Model ...")
    base_tt = TwoTowerModel(hidden_dim=64, depth=2, name="TwoTower_d2_h64")
    base_tt.prepare(train_pos, user_embs, item_embs, device=args.device)

    # 2. Models to evaluate
    models = [
        ("TwoTower_d2_h64", base_tt),
        ("TT_divloss_soft_jaccard_l0p1_s0p05", TwoTowerDivLoss(base_tt, div_loss_name="soft_jaccard", lambda_div=0.1, sigma=0.05)),
        ("TT_item_partition_n10", TwoTowerItemPartition(base_tt, n_trials=10)),
        ("TT_semantic_stratified_partition_n10", TwoTowerSemanticStratifiedPartition(base_tt, n_trials=10, n_clusters=20)),
        ("TT_multihead_stratified_partition_n10", TwoTowerMultiHeadStratifiedPartition(base_tt, n_trials=10, n_clusters=20)),
    ]

    all_results = {}
    for name, model in models:
        log.info(f"\n============================================================\nEvaluating Model: {name}\n============================================================")
        res = run_experiment_014(model, test_gt, train_pos, user_embs, item_embs, item2genres, iid2idx, args.device)
        all_results[name] = res

    # Save Results
    with open(OUT_DIR / "results_014.json", "w") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for name, res in all_results.items():
        m = res["mean"]
        rows.append({
            "model": name,
            "recall_cum": m["recall_cum"],
            "recall_avg": m["recall_avg"],
            "slate_precision_pct": m["slate_precision"],
            "gross_hits": m["gross_hits"],
            "hit": m["hit"],
            "ils": m["ils"],
            "hgts": m["hgts"],
            "hgc": m["hgc"],
            "diversity": m["diversity"],
        })
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUT_DIR / "results_014.csv", index=False)
    log.info(f"Saved CSV -> {OUT_DIR / 'results_014.csv'}")

    plot_plan_014_tradeoff(all_results, OUT_DIR)

    log.info("\n" + "="*70 + "\nPlan 014 Verification Summary\n" + "="*70)
    for name, res in all_results.items():
        m = res["mean"]
        log.info(
            f"{name:40s} | rc={m['recall_cum']:.4f} | ra={m['recall_avg']:.4f} | "
            f"SlatePrec={m['slate_precision']:.2f}% | GrossHits={m['gross_hits']:.2f} | "
            f"HGTS={m['hgts']:.4f} | HGC={m['hgc']:.2f}"
        )
    log.info("\n✅ Plan 014 evaluation completed successfully!")


if __name__ == "__main__":
    main()
