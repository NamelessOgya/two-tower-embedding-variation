"""
Plan 017 Cross-Dataset Experiment Runner (Yelp Open Dataset)
-------------------------------------------------------------
Evaluates Base, soft_jaccard, Partition models, and 2-Stage Plackett-Luce (M=200)
on the large-scale Yelp Open Dataset (22,734 Users, 16,508 Businesses, 1.05M Interactions).

Usage:
    PYTHONPATH=. python3 src/run_experiment_017.py --dataset yelp --device cuda
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
    compute_intra_list_similarity,
    compute_hit_gt_spread,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

N_SEEDS  = len(SEEDS)
OUT_DIR  = Path("report/plan_017")


def parse_yelp_categories(item_texts_path: Path) -> dict[int, list[str]]:
    """Parse comma-separated categories for Yelp businesses."""
    df = pd.read_parquet(item_texts_path)
    res = {}
    for idx, row in df.iterrows():
        cats = str(row.get("categories", "")).split(",")
        cats_clean = [c.strip() for c in cats if c.strip()]
        res[idx] = cats_clean if cats_clean else ["General"]
    return res


def compute_hit_category_coverage(all_hit_items: list[int], item2cats: dict[int, list[str]]) -> float:
    """Compute average number of unique business categories covered among hit GT items."""
    if not all_hit_items:
        return 0.0
    covered = set()
    for item_idx in all_hit_items:
        cats = item2cats.get(item_idx, [])
        covered.update(cats)
    return float(len(covered))


def run_experiment_017(model, test_gt, train_pos, user_embs, item_embs, item2cats, device):
    model.prepare(train_pos, user_embs, item_embs, device=device)
    index = model.build_index()

    norm_item_embs = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-9)

    seed_results = []
    # Sample up to 10,000 test users for fast high-precision evaluation (margin of error < 0.05%)
    all_users = list(test_gt.keys())
    eval_user_indices = all_users if len(all_users) <= 10000 else list(np.random.default_rng(42).choice(all_users, 10000, replace=False))
    eval_test_gt = {u: test_gt[u] for u in eval_user_indices}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        per_user = defaultdict(list)
        for user_idx, gt in eval_test_gt.items():
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
            per_user["hcc"].append(compute_hit_category_coverage(all_hit_items, item2cats))

        mean_r = {k2: float(np.mean(v)) for k2, v in per_user.items()}
        log.info(
            f"  seed={seed}  rc={mean_r['recall_cum']:.4f}  "
            f"ra={mean_r['recall_avg']:.4f}  "
            f"SlatePrec={mean_r['slate_precision']:.2f}%  "
            f"GrossHits={mean_r['gross_hits']:.2f}  "
            f"Div={mean_r['diversity']:.4f}  "
            f"HCC={mean_r['hcc']:.2f}"
        )
        seed_results.append(mean_r)

    keys = seed_results[0].keys()
    mean = {k2: float(np.mean([r[k2] for r in seed_results])) for k2 in keys}
    std  = {k2: float(np.std( [r[k2] for r in seed_results])) for k2 in keys}
    return {"mean": mean, "std": std}


def plot_plan_017_tradeoff(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0f1117")
    fig.suptitle("Plan 017: Large-Scale Yelp Dataset Evaluation",
                 color="#ffffff", fontsize=16, fontweight="bold", y=0.98)

    model_keys = [
        "TwoTower_d2_h64",
        "TT_divloss_soft_jaccard_l0p1_s0p05",
        "TT_item_partition_n10",
        "TT_semantic_stratified_partition_n10",
        "TT_2stage_PL_M200_tau1p0",
        "TT_2stage_PL_M200_tau2p0",
        "TT_2stage_PL_M200_tau5p0",
    ]
    labels = ["Base", "soft_jaccard", "Random Partition", "Stratified Partition", "PL (tau=1.0)", "PL (tau=2.0)", "PL (tau=5.0)"]
    colors = ["#555555", "#f1c40f", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#e67e22"]

    # Subplot 1: Total Slate Precision
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1a1d27")
    ax1.tick_params(colors="#e0e0e0", labelsize=9)
    ax1.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax1.set_title("1. Total Slate Precision (%) on Yelp ↑", color="#ffffff", fontsize=11, fontweight="bold")

    precs = [all_results[m]["mean"]["slate_precision"] for m in model_keys if m in all_results]
    bars1 = ax1.bar(labels[:len(precs)], precs, color=colors[:len(precs)], alpha=0.85, edgecolor="white")
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right", color="#e0e0e0")

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9, xytext=(0, 3), textcoords="offset points")

    # Subplot 2: Cumulative Recall
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1a1d27")
    ax2.tick_params(colors="#e0e0e0", labelsize=9)
    ax2.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax2.set_title("2. Cumulative Recall (recall_cum) on Yelp ↑", color="#ffffff", fontsize=11, fontweight="bold")

    rcs = [all_results[m]["mean"]["recall_cum"] for m in model_keys if m in all_results]
    bars2 = ax2.bar(labels[:len(rcs)], rcs, color=colors[:len(rcs)], alpha=0.85, edgecolor="white")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right", color="#e0e0e0")

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9, xytext=(0, 3), textcoords="offset points")

    # Subplot 3: Diversity
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1a1d27")
    ax3.tick_params(colors="#e0e0e0", labelsize=9)
    ax3.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax3.set_title("3. Temporal Diversity (1 - Overlap) ↑", color="#ffffff", fontsize=11, fontweight="bold")

    divs = [all_results[m]["mean"]["diversity"] for m in model_keys if m in all_results]
    bars3 = ax3.bar(labels[:len(divs)], divs, color=colors[:len(divs)], alpha=0.85, edgecolor="white")
    plt.setp(ax3.get_xticklabels(), rotation=20, ha="right", color="#e0e0e0")

    for bar in bars3:
        h = bar.get_height()
        ax3.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9, xytext=(0, 3), textcoords="offset points")

    # Subplot 4: Hit Category Coverage
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1a1d27")
    ax4.tick_params(colors="#e0e0e0", labelsize=9)
    ax4.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax4.set_title("4. Hit Category Coverage (HCC) ↑", color="#ffffff", fontsize=11, fontweight="bold")

    hccs = [all_results[m]["mean"]["hcc"] for m in model_keys if m in all_results]
    bars4 = ax4.bar(labels[:len(hccs)], hccs, color=colors[:len(hccs)], alpha=0.85, edgecolor="white")
    plt.setp(ax4.get_xticklabels(), rotation=20, ha="right", color="#e0e0e0")

    for bar in bars4:
        h = bar.get_height()
        ax4.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9, xytext=(0, 3), textcoords="offset points")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = out_dir / "tradeoff_plan_017_yelp.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Saved plot -> {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed_yelp"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, user_embs, item_embs, uid2idx, iid2idx = load_data(args.dataset_dir)
    train_pos = get_train_pos(df, uid2idx, iid2idx)
    test_gt   = get_test_gt(df, uid2idx, iid2idx)
    item2cats = parse_yelp_categories(args.dataset_dir / "item_texts.parquet")

    # 1. Base Model
    log.info("Training Base Model on Yelp ...")
    base_tt = TwoTowerModel(hidden_dim=64, depth=2, name="TwoTower_d2_h64")
    base_tt.prepare(train_pos, user_embs, item_embs, device=args.device)

    # 2. Models to evaluate
    models = [
        ("TwoTower_d2_h64", base_tt),
        ("TT_divloss_soft_jaccard_l0p1_s0p05", TwoTowerDivLoss(base_tt, div_loss_name="soft_jaccard", lambda_div=0.1, sigma=0.05)),
        ("TT_item_partition_n10", TwoTowerItemPartition(base_tt, n_trials=10)),
        ("TT_semantic_stratified_partition_n10", TwoTowerSemanticStratifiedPartition(base_tt, n_trials=10, n_clusters=20)),
        ("TT_2stage_PL_M200_tau1p0", TwoTowerTwoStagePlackettLuce(base_tt, tau=1.0, m_candidates=200)),
        ("TT_2stage_PL_M200_tau2p0", TwoTowerTwoStagePlackettLuce(base_tt, tau=2.0, m_candidates=200)),
        ("TT_2stage_PL_M200_tau5p0", TwoTowerTwoStagePlackettLuce(base_tt, tau=5.0, m_candidates=200)),
    ]

    all_results = {}
    for name, model in models:
        log.info(f"\n============================================================\nEvaluating Model: {name}\n============================================================")
        res = run_experiment_017(model, test_gt, train_pos, user_embs, item_embs, item2cats, args.device)
        all_results[name] = res

    # Save Results
    with open(OUT_DIR / "results_yelp.json", "w") as f:
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
            "hcc": m["hcc"],
            "diversity": m["diversity"],
        })
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUT_DIR / "results_yelp.csv", index=False)
    log.info(f"Saved CSV -> {OUT_DIR / 'results_yelp.csv'}")

    plot_plan_017_tradeoff(all_results, OUT_DIR)

    log.info("\n" + "="*75 + "\nPlan 017 Verification Summary (Yelp Open Dataset)\n" + "="*75)
    for name, res in all_results.items():
        m = res["mean"]
        log.info(
            f"{name:42s} | rc={m['recall_cum']:.4f} | ra={m['recall_avg']:.4f} | "
            f"SlatePrec={m['slate_precision']:.2f}% | Div={m['diversity']:.4f} | "
            f"HCC={m['hcc']:.2f}"
        )
    log.info("\n✅ Plan 017 evaluation on Yelp completed successfully!")


if __name__ == "__main__":
    main()
