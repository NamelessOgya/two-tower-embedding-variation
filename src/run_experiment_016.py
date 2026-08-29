"""
Plan 016 Experiment Runner: Two-Stage Plackett-Luce Probabilistic Ranking Evaluation
-------------------------------------------------------------------------------------
Evaluates the 2-Stage Plackett-Luce model (Stage 1 FAISS M=200 -> Stage 2 Gumbel-Top-K Sampling)
across temperatures tau in [0.2, 0.5, 1.0, 2.0, 5.0] against Base, soft_jaccard, Random Partition, and Stratified Partition.

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
    PYTHONPATH=. python3 src/run_experiment_016.py --device cuda
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
from src.model.models_014 import TwoTowerSemanticStratifiedPartition
from src.model.models_016 import TwoTowerTwoStagePlackettLuce
from src.run_experiment_013 import (
    parse_item_genres,
    compute_intra_list_similarity,
    compute_hit_gt_spread,
    compute_hit_genre_coverage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

N_SEEDS  = len(SEEDS)
OUT_DIR  = Path("report/plan_016")
DATA_DIR = Path("data/processed/movielens")


def run_experiment_016(model, test_gt, train_pos, user_embs, item_embs, item2genres, iid2idx, device):
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

            slate_prec = (ra * gt_len / float(K)) * 100.0
            gross_hits = ra * gt_len * float(N_TRIALS)

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
            f"Div={mean_r['diversity']:.4f}  "
            f"HGTS={mean_r['hgts']:.4f}"
        )
        seed_results.append(mean_r)

    keys = seed_results[0].keys()
    mean = {k2: float(np.mean([r[k2] for r in seed_results])) for k2 in keys}
    std  = {k2: float(np.std( [r[k2] for r in seed_results])) for k2 in keys}
    return {"mean": mean, "std": std}


def plot_plan_016_tradeoff(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0f1117")
    fig.suptitle("Plan 016: Two-Stage Plackett-Luce Probabilistic Ranking Evaluation",
                 color="#ffffff", fontsize=16, fontweight="bold", y=0.98)

    # Diversity vs recall_cum (Subplot 1)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1a1d27")
    ax1.tick_params(colors="#e0e0e0", labelsize=9)
    ax1.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax1.set_title("1. Diversity (1-Overlap) vs Cumulative Recall (recall_cum)", color="#ffffff", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Diversity (1 - temporal_overlap) ↑", color="#e0e0e0", fontsize=10)
    ax1.set_ylabel("Cumulative Recall (recall_cum) ↑", color="#e0e0e0", fontsize=10)

    # Diversity vs Slate Precision (Subplot 2)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1a1d27")
    ax2.tick_params(colors="#e0e0e0", labelsize=9)
    ax2.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax2.set_title("2. Diversity (1-Overlap) vs Total Slate Precision (%)", color="#ffffff", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Diversity (1 - temporal_overlap) ↑", color="#e0e0e0", fontsize=10)
    ax2.set_ylabel("Total Slate Precision (%) ↑", color="#e0e0e0", fontsize=10)

    # Plot PL tau sweep
    pl_taus = [0.2, 0.5, 1.0, 2.0, 5.0]
    pl_models = [f"TT_2stage_PL_M200_tau{str(t).replace('.', 'p')}" for t in pl_taus]

    pl_divs = [all_results[m]["mean"]["diversity"] for m in pl_models if m in all_results]
    pl_rcs  = [all_results[m]["mean"]["recall_cum"] for m in pl_models if m in all_results]
    pl_precs= [all_results[m]["mean"]["slate_precision"] for m in pl_models if m in all_results]

    ax1.plot(pl_divs, pl_rcs, "o-", color="#e74c3c", lw=2, ms=8, label="2-Stage Plackett-Luce (M=200, tau sweep)")
    for i, t in enumerate(pl_taus):
        if i < len(pl_divs):
            ax1.annotate(f"tau={t}", (pl_divs[i], pl_rcs[i]), xytext=(5, 5), textcoords="offset points", color="#f39c12", fontsize=9, fontweight="bold")

    ax2.plot(pl_divs, pl_precs, "o-", color="#e74c3c", lw=2, ms=8, label="2-Stage Plackett-Luce (M=200, tau sweep)")
    for i, t in enumerate(pl_taus):
        if i < len(pl_divs):
            ax2.annotate(f"tau={t}", (pl_divs[i], pl_precs[i]), xytext=(5, 5), textcoords="offset points", color="#f39c12", fontsize=9, fontweight="bold")

    # Plot benchmark baselines
    benchmarks = [
        ("TwoTower_d2_h64", "Base (no-div)", "#555555", "s"),
        ("TT_divloss_soft_jaccard_l0p1_s0p05", "soft_jaccard", "#f1c40f", "^"),
        ("TT_item_partition_n10", "Random Partition", "#3498db", "D"),
        ("TT_semantic_stratified_partition_n10", "Semantic Stratified", "#2ecc71", "*"),
    ]

    for name, label, color, marker in benchmarks:
        if name in all_results:
            div = all_results[name]["mean"]["diversity"]
            rc  = all_results[name]["mean"]["recall_cum"]
            prec = all_results[name]["mean"]["slate_precision"]

            ax1.scatter([div], [rc], color=color, marker=marker, s=120, label=label, zorder=5)
            ax2.scatter([div], [prec], color=color, marker=marker, s=120, label=label, zorder=5)

    ax1.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=9)
    ax2.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=9)

    # Subplot 3: Bar chart for Slate Precision
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1a1d27")
    ax3.tick_params(colors="#e0e0e0", labelsize=9)
    ax3.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax3.set_title("3. Total Slate Precision Comparison (%)", color="#ffffff", fontsize=11, fontweight="bold")

    model_keys = ["TwoTower_d2_h64", "TT_divloss_soft_jaccard_l0p1_s0p05", "TT_item_partition_n10", "TT_semantic_stratified_partition_n10", "TT_2stage_PL_M200_tau0p5", "TT_2stage_PL_M200_tau1p0"]
    model_labels = ["Base", "soft_jaccard", "Random Partition", "Stratified Partition", "PL (tau=0.5)", "PL (tau=1.0)"]
    bar_colors = ["#555555", "#f1c40f", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    bar_vals = [all_results[m]["mean"]["slate_precision"] for m in model_keys if m in all_results]
    bars = ax3.bar(model_labels[:len(bar_vals)], bar_vals, color=bar_colors[:len(bar_vals)], alpha=0.85, edgecolor="white")
    plt.setp(ax3.get_xticklabels(), rotation=15, ha="right", color="#e0e0e0")

    for bar in bars:
        h = bar.get_height()
        ax3.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9, xytext=(0, 3), textcoords="offset points")

    # Subplot 4: Bar chart for Cumulative Recall
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1a1d27")
    ax4.tick_params(colors="#e0e0e0", labelsize=9)
    ax4.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax4.set_title("4. Cumulative Recall Comparison (recall_cum)", color="#ffffff", fontsize=11, fontweight="bold")

    bar_rcs = [all_results[m]["mean"]["recall_cum"] for m in model_keys if m in all_results]
    bars4 = ax4.bar(model_labels[:len(bar_rcs)], bar_rcs, color=bar_colors[:len(bar_rcs)], alpha=0.85, edgecolor="white")
    plt.setp(ax4.get_xticklabels(), rotation=15, ha="right", color="#e0e0e0")

    for bar in bars4:
        h = bar.get_height()
        ax4.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9, xytext=(0, 3), textcoords="offset points")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = out_dir / "tradeoff_plan_016.png"
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
    ]

    # Add Plackett-Luce tau sweep (M=200)
    for tau in [0.2, 0.5, 1.0, 2.0, 5.0]:
        models.append((
            f"TT_2stage_PL_M200_tau{str(tau).replace('.', 'p')}",
            TwoTowerTwoStagePlackettLuce(base_tt, tau=tau, m_candidates=200)
        ))

    all_results = {}
    for name, model in models:
        log.info(f"\n============================================================\nEvaluating Model: {name}\n============================================================")
        res = run_experiment_016(model, test_gt, train_pos, user_embs, item_embs, item2genres, iid2idx, args.device)
        all_results[name] = res

    # Save Results
    with open(OUT_DIR / "results_016.json", "w") as f:
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
    df_metrics.to_csv(OUT_DIR / "results_016.csv", index=False)
    log.info(f"Saved CSV -> {OUT_DIR / 'results_016.csv'}")

    plot_plan_016_tradeoff(all_results, OUT_DIR)

    log.info("\n" + "="*75 + "\nPlan 016 Verification Summary\n" + "="*75)
    for name, res in all_results.items():
        m = res["mean"]
        log.info(
            f"{name:45s} | rc={m['recall_cum']:.4f} | ra={m['recall_avg']:.4f} | "
            f"SlatePrec={m['slate_precision']:.2f}% | Div={m['diversity']:.4f} | "
            f"HGTS={m['hgts']:.4f} | HGC={m['hgc']:.2f}"
        )
    log.info("\n✅ Plan 016 evaluation completed successfully!")


if __name__ == "__main__":
    main()
