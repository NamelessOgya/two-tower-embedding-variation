"""
Plan 020 Runner: Next-Gen LLM Embedding Benchmark (Qwen3 / Qwen2 vs. mE5)
-------------------------------------------------------------------------
Compares Embedding Architectures and Scales on downstream TwoTowerLogQInfoNCE:
  - mE5-small (384-dim, ~118M params, RoBERTa)
  - mE5-base  (768-dim, ~278M params, RoBERTa)
  - mE5-large (1024-dim, ~560M params, RoBERTa)
  - Qwen3-0.6B (1024-dim, ~600M params, Qwen3 LLM)
  - GTE-Qwen2-1.5B (1536-dim, ~1.5B params, Qwen2 LLM)

Evaluates 5 seeds with single-trial accuracy and aggregate diversity metrics.
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

from src.model.models_018 import TwoTowerLogQInfoNCE
from src.run_experiment_018 import compute_single_trial_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEEDS = [42, 43, 44, 45, 46]
K = 10


def load_dataset(data_dir: Path):
    log.info(f"Loading data from {data_dir} ...")
    interactions = pd.read_parquet(data_dir / "interactions.parquet")
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

    return train_pos, test_pos


def evaluate_scale_5seeds(
    scale_name: str,
    user_embs: np.ndarray,
    item_embs: np.ndarray,
    train_pos: dict,
    test_pos: dict,
    device: str = "cuda",
) -> dict[str, Any]:
    n_users = len(user_embs)
    n_items = len(item_embs)
    in_dim = user_embs.shape[1]

    train_item_counts = np.zeros(n_items, dtype=np.int64)
    for items in train_pos.values():
        for iid in items:
            if 0 <= iid < n_items:
                train_item_counts[iid] += 1

    seed_results = []
    for seed in SEEDS:
        log.info(f"--- Running {scale_name} ({in_dim}d) (Seed {seed}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = TwoTowerLogQInfoNCE(
            hidden_dim=64,
            depth=2,
            epochs=25,
            batch_size=2048,
            lr=2e-3,
            tau=0.07,
            alpha=1.0,
            name=f"TT_LogQ_{scale_name}",
        )

        model.prepare(train_pos, user_embs, item_embs, device=device)
        index = model.build_index()

        all_user_recs = []
        rng = np.random.default_rng(seed)
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
        "model": scale_name,
        "in_dim": in_dim,
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


def plot_plan_020_results(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Plan 020: Next-Gen LLM Embedding Benchmark (MovieLens 1M)\n(5 Seeds Mean ± Std Error Bars)", fontsize=15, fontweight='bold')

    palette = {
        "mE5-small (384d)": "#3498db",
        "mE5-base (768d)": "#2980b9",
        "mE5-large (1024d)": "#1f618d",
        "Qwen3-0.6B (1024d)": "#e67e22",
        "GTE-Qwen2-1.5B (1536d)": "#d35400",
    }
    colors = [palette.get(m, "#2ecc71") for m in df["model"]]

    # (a) Recall@10
    ax = axes[0, 0]
    x_pos = np.arange(len(df))
    ax.bar(
        x_pos,
        df["recall_10_mean"],
        yerr=df["recall_10_std"],
        color=colors,
        alpha=0.85,
        capsize=5,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["model"], rotation=15, ha='right', fontweight='bold', fontsize=9)
    ax.set_title("(a) Recall@10 vs. Embedding Model ↑", fontweight='bold', fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(df["recall_10_mean"]):
        ax.text(i, v / 2, f"{v:.4f}", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # (b) Precision@10 (%)
    ax = axes[0, 1]
    ax.bar(
        x_pos,
        df["precision_10_mean"],
        yerr=df["precision_10_std"],
        color=colors,
        alpha=0.85,
        capsize=5,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["model"], rotation=15, ha='right', fontweight='bold', fontsize=9)
    ax.set_title("(b) Precision@10 (%) vs. Embedding Model ↑", fontweight='bold', fontsize=11)
    ax.set_ylabel("Precision@10 (%)", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(df["precision_10_mean"]):
        ax.text(i, v / 2, f"{v:.2f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # (c) Catalog Coverage@10 (%)
    ax = axes[1, 0]
    ax.bar(
        x_pos,
        df["coverage_10_mean"],
        yerr=df["coverage_10_std"],
        color=colors,
        alpha=0.85,
        capsize=5,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["model"], rotation=15, ha='right', fontweight='bold', fontsize=9)
    ax.set_title("(c) Catalog Coverage@10 (%) vs. Embedding Model", fontweight='bold', fontsize=11)
    ax.set_ylabel("Coverage@10 (%)", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(df["coverage_10_mean"]):
        ax.text(i, v / 2, f"{v:.2f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # (d) Trade-off: Recall@10 vs. Catalog Coverage@10 (%)
    ax = axes[1, 1]
    for i, row in df.iterrows():
        c = colors[i]
        ax.errorbar(
            row["coverage_10_mean"],
            row["recall_10_mean"],
            xerr=row["coverage_10_std"],
            yerr=row["recall_10_std"],
            fmt='o',
            color=c,
            ecolor=c,
            elinewidth=2,
            capsize=5,
            markersize=10,
            label=row["model"],
        )
    ax.set_title("(d) Trade-off: Recall@10 vs. Catalog Coverage@10 (%)", fontweight='bold', fontsize=11)
    ax.set_xlabel("Catalog Coverage@10 (%) [Higher is Better →]", fontsize=10)
    ax.set_ylabel("Recall@10 [Higher is Better →]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    log.info(f"Saved plot with error bars to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/movielens"))
    parser.add_argument("--scales-dir", type=Path, default=Path("data/processed_scales"))
    parser.add_argument("--out-dir", type=Path, default=Path("report/plan_020"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_pos, test_pos = load_dataset(args.data_dir)

    models_to_test = [
        ("mE5-small (384d)", args.scales_dir / "user_embeddings_small.npy", args.scales_dir / "item_embeddings_small.npy"),
        ("mE5-base (768d)",  args.data_dir / "user_embeddings.npy",       args.data_dir / "item_embeddings.npy"),
        ("mE5-large (1024d)", args.scales_dir / "user_embeddings_large.npy", args.scales_dir / "item_embeddings_large.npy"),
        ("Qwen3-0.6B (1024d)", args.scales_dir / "user_embeddings_qwen3_0p6b.npy", args.scales_dir / "item_embeddings_qwen3_0p6b.npy"),
        ("GTE-Qwen2-1.5B (1536d)", args.scales_dir / "user_embeddings_gte_qwen2_1p5b.npy", args.scales_dir / "item_embeddings_gte_qwen2_1p5b.npy"),
    ]

    all_summaries = []
    for scale_name, u_path, i_path in models_to_test:
        if not u_path.exists() or not i_path.exists():
            log.warning(f"Skipping {scale_name}: files {u_path} or {i_path} not found.")
            continue
        log.info(f"Loading embeddings for {scale_name}: {u_path} & {i_path}")
        user_embs = np.load(u_path).astype(np.float32)
        item_embs = np.load(i_path).astype(np.float32)

        summary = evaluate_scale_5seeds(
            scale_name, user_embs, item_embs, train_pos, test_pos, device=args.device
        )
        all_summaries.append(summary)

    df_results = pd.DataFrame(all_summaries)
    csv_path = args.out_dir / "results_020.csv"
    df_results.to_csv(csv_path, index=False)
    log.info(f"Saved Plan 020 results CSV to {csv_path}")

    plot_path = args.out_dir / "tradeoff_plan_020.png"
    plot_plan_020_results(df_results, plot_path)


if __name__ == "__main__":
    main()
