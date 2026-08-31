"""
Plan 018 Runner: Two-Tower Training-Time Popularity De-biasing Benchmark
-----------------------------------------------------------------------
Runs 5-seed evaluation on MovieLens 1M and Yelp 10-Core.
Evaluates single-trial recommendation metrics (Accuracy vs. Aggregate Diversity & Bias).

Metrics evaluated:
  - Accuracy: recall_10, precision_10 (%), ndcg_10, hit_10 (%)
  - Diversity & Bias: coverage_10 (%), gini_index, entropy, longtail_coverage (%)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.model.models_018 import (
    TwoTowerBPRBase,
    TwoTowerLogQInfoNCE,
    TwoTowerPopNegativeBPR,
    TwoTowerUniformityLoss,
    TwoTowerAdaptiveTauInfoNCE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEEDS = [42, 43, 44, 45, 46]
K = 10


# ── Metrics Calculator ────────────────────────────────────────────────────────

def compute_single_trial_metrics(
    all_user_recs: list[list[int]],
    test_pos_per_user: dict[int, np.ndarray],
    n_total_items: int,
    train_item_counts: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    recalls = []
    precisions = []
    ndcgs = []
    hits = []

    rec_counts = np.zeros(n_total_items, dtype=np.int64)

    for u_idx, rec_items in enumerate(all_user_recs):
        gt = set(test_pos_per_user.get(u_idx, []))
        if len(gt) == 0:
            continue
        hits_k = [1 if item in gt else 0 for item in rec_items[:k]]
        n_hits = sum(hits_k)

        recalls.append(n_hits / len(gt))
        precisions.append(n_hits / k)
        hits.append(1.0 if n_hits > 0 else 0.0)

        # NDCG@K
        dcg = sum([hits_k[i] / np.log2(i + 2) for i in range(len(hits_k))])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gt), k))])
        ndcgs.append(dcg / max(idcg, 1e-9))

        for item in rec_items[:k]:
            if 0 <= item < n_total_items:
                rec_counts[item] += 1

    recall_10 = float(np.mean(recalls)) if recalls else 0.0
    precision_10 = float(np.mean(precisions)) * 100.0 if precisions else 0.0
    ndcg_10 = float(np.mean(ndcgs)) if ndcgs else 0.0
    hit_10 = float(np.mean(hits)) * 100.0 if hits else 0.0

    # Catalog Coverage@10 (%)
    unique_recommended = np.sum(rec_counts > 0)
    coverage_10 = float(unique_recommended / n_total_items) * 100.0

    # Gini Index (0: 完全均等, 1: 極度集中)
    sorted_counts = np.sort(rec_counts)
    n = len(sorted_counts)
    total = np.sum(sorted_counts)
    if total > 0:
        index = np.arange(1, n + 1)
        gini = float(np.sum((2 * index - n - 1) * sorted_counts) / (n * total))
    else:
        gini = 1.0

    # Shannon Entropy (bits)
    if total > 0:
        p = sorted_counts[sorted_counts > 0] / total
        entropy = float(-np.sum(p * np.log2(p)))
    else:
        entropy = 0.0

    # Long-tail Coverage (%)
    # 学習時出現回数が下位 80% のアイテム
    tail_threshold = np.percentile(train_item_counts, 80)
    tail_items_mask = (train_item_counts <= tail_threshold)
    n_tail_items = np.sum(tail_items_mask)
    if n_tail_items > 0:
        tail_rec_mask = (rec_counts > 0) & tail_items_mask
        longtail_coverage = float(np.sum(tail_rec_mask) / n_tail_items) * 100.0
    else:
        longtail_coverage = 0.0

    return {
        "recall_10": recall_10,
        "precision_10": precision_10,
        "ndcg_10": ndcg_10,
        "hit_10": hit_10,
        "coverage_10": coverage_10,
        "gini_index": gini,
        "entropy": entropy,
        "longtail_coverage": longtail_coverage,
    }


# ── Load Dataset ──────────────────────────────────────────────────────────────

def load_data(data_dir: Path):
    log.info(f"Loading data from {data_dir} ...")
    interactions = pd.read_parquet(data_dir / "interactions.parquet")
    user_embeddings = np.load(data_dir / "user_embeddings.npy").astype(np.float32)
    item_embeddings = np.load(data_dir / "item_embeddings.npy").astype(np.float32)
    user_id_map = pd.read_parquet(data_dir / "user_id_map.parquet")
    item_id_map = pd.read_parquet(data_dir / "item_id_map.parquet")

    uid2idx = dict(zip(user_id_map["user_id"], user_id_map["index"]))
    iid2idx = dict(zip(item_id_map["item_id"], item_id_map["index"]))

    train_df = interactions[(interactions["split"] == "train") & (interactions["is_positive"])]
    test_df = interactions[(interactions["split"] == "test") & (interactions["is_positive"])]

    train_pos: dict[int, np.ndarray] = {}
    for uid, grp in train_df.groupby("user_id"):
        uidx = uid2idx.get(uid)
        if uidx is None:
            continue
        idxs = [iid2idx[iid] for iid in grp["item_id"] if iid in iid2idx]
        if idxs:
            train_pos[uidx] = np.array(idxs, dtype=np.int64)

    test_pos: dict[int, np.ndarray] = {}
    for uid, grp in test_df.groupby("user_id"):
        uidx = uid2idx.get(uid)
        if uidx is None:
            continue
        idxs = [iid2idx[iid] for iid in grp["item_id"] if iid in iid2idx]
        if idxs:
            test_pos[uidx] = np.array(idxs, dtype=np.int64)

    n_total_items = len(item_embeddings)
    train_item_counts = np.zeros(n_total_items, dtype=np.int64)
    for items in train_pos.values():
        for iid in items:
            if 0 <= iid < n_total_items:
                train_item_counts[iid] += 1

    log.info(f"Loaded: Users={len(user_embeddings)}, Items={len(item_embeddings)}, TrainUsers={len(train_pos)}, TestUsers={len(test_pos)}")
    return train_pos, test_pos, user_embeddings, item_embeddings, train_item_counts


# ── Evaluate Single Model on 5 Seeds ──────────────────────────────────────────

def evaluate_model_5seeds(
    model_fn,
    model_name: str,
    train_pos: dict,
    test_pos: dict,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    train_item_counts: np.ndarray,
    device: str = "cuda",
) -> dict[str, Any]:
    n_users = len(user_embeddings)
    n_items = len(item_embeddings)
    test_user_indices = sorted(test_pos.keys())

    seed_results = []

    for seed in SEEDS:
        log.info(f"--- Running {model_name} (Seed {seed}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_fn()
        model.prepare(train_pos, user_embeddings, item_embeddings, device=device)
        index = model.build_index()

        all_user_recs = []
        rng = np.random.default_rng(seed)

        # Batch retrieval across all test users
        for u_idx in range(n_users):
            if u_idx in test_pos:
                rec = model.recommend(u_idx, trial=0, rng=rng, index=index, k=K)
                all_user_recs.append(rec)
            else:
                all_user_recs.append([])

        m = compute_single_trial_metrics(
            all_user_recs, test_pos, n_items, train_item_counts, k=K
        )
        seed_results.append(m)
        log.info(f"  [Seed {seed}] Recall@10: {m['recall_10']:.4f}, Prec@10: {m['precision_10']:.2f}%, Coverage@10: {m['coverage_10']:.2f}%, Gini: {m['gini_index']:.4f}")

    df_seeds = pd.DataFrame(seed_results)
    summary = {
        "model": model_name,
        "recall_10_mean": df_seeds["recall_10"].mean(),
        "recall_10_std": df_seeds["recall_10"].std(),
        "precision_10_mean": df_seeds["precision_10"].mean(),
        "precision_10_std": df_seeds["precision_10"].std(),
        "ndcg_10_mean": df_seeds["ndcg_10"].mean(),
        "ndcg_10_std": df_seeds["ndcg_10"].std(),
        "hit_10_mean": df_seeds["hit_10"].mean(),
        "hit_10_std": df_seeds["hit_10"].std(),
        "coverage_10_mean": df_seeds["coverage_10"].mean(),
        "coverage_10_std": df_seeds["coverage_10"].std(),
        "gini_index_mean": df_seeds["gini_index"].mean(),
        "gini_index_std": df_seeds["gini_index"].std(),
        "entropy_mean": df_seeds["entropy"].mean(),
        "entropy_std": df_seeds["entropy"].std(),
        "longtail_coverage_mean": df_seeds["longtail_coverage"].mean(),
        "longtail_coverage_std": df_seeds["longtail_coverage"].std(),
    }
    return summary


# ── Plot Results with Error Bars ──────────────────────────────────────────────

def plot_plan_018_results(df: pd.DataFrame, out_path: Path, dataset_name: str = "MovieLens 1M"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Plan 018: Training-Time De-biasing Benchmark ({dataset_name})\n(5 Seeds Mean ± Std Error Bars)", fontsize=15, fontweight='bold')

    colors = ["#7f8c8d", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c"]

    # (a) Tradeoff: Recall@10 vs. Catalog Coverage@10 (%)
    ax = axes[0, 0]
    for i, row in df.iterrows():
        c = colors[i % len(colors)]
        ax.errorbar(
            row["coverage_10_mean"],
            row["recall_10_mean"],
            xerr=row["coverage_10_std"],
            yerr=row["recall_10_std"],
            fmt='o',
            color=c,
            ecolor=c,
            elinewidth=2,
            capsize=4,
            capthick=1.5,
            markersize=9,
            label=row["model"],
        )
    ax.set_title("(a) Trade-off: Recall@10 vs. Catalog Coverage@10 (%)", fontweight='bold', fontsize=11)
    ax.set_xlabel("Catalog Coverage@10 (%) [Higher is Better →]", fontsize=10)
    ax.set_ylabel("Recall@10 [Higher is Better →]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # (b) Tradeoff: Precision@10 (%) vs. Gini Index
    ax = axes[0, 1]
    for i, row in df.iterrows():
        c = colors[i % len(colors)]
        ax.errorbar(
            row["gini_index_mean"],
            row["precision_10_mean"],
            xerr=row["gini_index_std"],
            yerr=row["precision_10_std"],
            fmt='s',
            color=c,
            ecolor=c,
            elinewidth=2,
            capsize=4,
            capthick=1.5,
            markersize=9,
            label=row["model"],
        )
    ax.set_title("(b) Precision@10 (%) vs. Gini Index", fontweight='bold', fontsize=11)
    ax.set_xlabel("Gini Index [Lower is Fairer ←]", fontsize=10)
    ax.set_ylabel("Precision@10 (%) [Higher is Better →]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # (c) Catalog Coverage@10 (%) vs. Long-tail Coverage@10 (%)
    ax = axes[1, 0]
    for i, row in df.iterrows():
        c = colors[i % len(colors)]
        ax.errorbar(
            row["coverage_10_mean"],
            row["longtail_coverage_mean"],
            xerr=row["coverage_10_std"],
            yerr=row["longtail_coverage_std"],
            fmt='^',
            color=c,
            ecolor=c,
            elinewidth=2,
            capsize=4,
            capthick=1.5,
            markersize=9,
            label=row["model"],
        )
    ax.set_title("(c) Overall Coverage vs. Long-tail Coverage (%)", fontweight='bold', fontsize=11)
    ax.set_xlabel("Catalog Coverage@10 (%)", fontsize=10)
    ax.set_ylabel("Long-tail Coverage@10 (%) [Higher is Better →]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # (d) Accuracy-Diversity Summary: NDCG@10 vs. Shannon Entropy
    ax = axes[1, 1]
    for i, row in df.iterrows():
        c = colors[i % len(colors)]
        ax.errorbar(
            row["entropy_mean"],
            row["ndcg_10_mean"],
            xerr=row["entropy_std"],
            yerr=row["ndcg_10_std"],
            fmt='D',
            color=c,
            ecolor=c,
            elinewidth=2,
            capsize=4,
            capthick=1.5,
            markersize=9,
            label=row["model"],
        )
    ax.set_title("(d) NDCG@10 vs. Shannon Entropy", fontweight='bold', fontsize=11)
    ax.set_xlabel("Shannon Entropy (bits) [Higher is More Diverse →]", fontsize=10)
    ax.set_ylabel("NDCG@10 [Higher is Better →]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log.info(f"Saved plot with error bars to {out_path}")


# ── Main Runner ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="data/processed/movielens")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="report/plan_018")
    parser.add_argument("--dataset-name", type=str, default="MovieLens 1M")
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pos, test_pos, user_embs, item_embs, train_item_counts = load_data(data_dir)

    models_to_test = [
        # 1. Base
        ("TT_BPR_base", lambda: TwoTowerBPRBase(hidden_dim=64, depth=2, epochs=25, batch_size=2048, lr=2e-3)),
        # 2. Log-Q InfoNCE (Google RecSys'19)
        ("TT_LogQ_InfoNCE_a1p0", lambda: TwoTowerLogQInfoNCE(hidden_dim=64, depth=2, epochs=25, batch_size=2048, lr=2e-3, tau=0.07, alpha=1.0)),
        ("TT_LogQ_InfoNCE_a0p5", lambda: TwoTowerLogQInfoNCE(hidden_dim=64, depth=2, epochs=25, batch_size=2048, lr=2e-3, tau=0.07, alpha=0.5)),
        # 3. Popularity Negative Sampling (Google WWW'20)
        ("TT_PopNeg_b0p75", lambda: TwoTowerPopNegativeBPR(hidden_dim=64, depth=2, epochs=25, batch_size=2048, lr=2e-3, beta=0.75)),
        ("TT_PopNeg_b0p50", lambda: TwoTowerPopNegativeBPR(hidden_dim=64, depth=2, epochs=25, batch_size=2048, lr=2e-3, beta=0.50)),
        # 4. Uniformity Regularization (ICML'20 / KDD'22)
        ("TT_Uniformity_l1p0", lambda: TwoTowerUniformityLoss(hidden_dim=64, depth=2, epochs=25, batch_size=2048, lr=2e-3, lambda_uni=1.0)),
        ("TT_Uniformity_l0p2", lambda: TwoTowerUniformityLoss(hidden_dim=64, depth=2, epochs=25, batch_size=2048, lr=2e-3, lambda_uni=0.2)),
        # 5. Adaptive Tau InfoNCE
        ("TT_AdaptTau_g1p0", lambda: TwoTowerAdaptiveTauInfoNCE(hidden_dim=64, depth=2, epochs=25, batch_size=2048, lr=2e-3, tau_base=0.07, gamma=1.0)),
    ]

    all_summaries = []
    for model_name, model_fn in models_to_test:
        summary = evaluate_model_5seeds(
            model_fn, model_name, train_pos, test_pos, user_embs, item_embs, train_item_counts, device=args.device
        )
        all_summaries.append(summary)

    df_results = pd.DataFrame(all_summaries)
    csv_name = "results_018_movielens.csv" if "movielens" in str(data_dir).lower() else "results_018_yelp.csv"
    csv_path = out_dir / csv_name
    df_results.to_csv(csv_path, index=False)
    log.info(f"Saved results CSV to {csv_path}")

    plot_name = "tradeoff_plan_018_movielens.png" if "movielens" in str(data_dir).lower() else "tradeoff_plan_018_yelp.png"
    plot_path = out_dir / plot_name
    plot_plan_018_results(df_results, plot_path, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
