"""
Plan 013 Experiment Runner: Hypothesis Verification for Item Partition
-----------------------------------------------------------------------
仮説「全アイテム検索は同質アイテムで Top-10 の枠が占領されているのに対し、
Item Partition は枠の占領を解除し、多様な Ground Truth（正解群）を効率よく救い出している」
を quantitative metrics (ILS, HGTS, HGC) と 4-panel プロットで検証・証明する。

Usage:
    PYTHONPATH=. python3 src/run_experiment_013.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
from src.model.models_007 import TwoTowerModel, TwoTowerPostNoise
from src.model.models_008 import TwoTowerDivLoss
from src.model.models_012 import TwoTowerItemPartition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BEST_DEPTH      = 2
BEST_HIDDEN_DIM = 64
N_SEEDS         = len(SEEDS)

OUT_DIR  = Path("report/plan_013")
DATA_DIR = Path("data/processed/movielens")


def parse_item_genres(item_texts_path: Path) -> dict[int, set[str]]:
    """item_texts.parquet から各アイテムのジャンル集合を取得する。"""
    df = pd.read_parquet(item_texts_path)
    item2genres = {}
    for _, row in df.iterrows():
        iid = int(row["item_id"])
        text = str(row["item_text"])
        m = re.search(r"Genres:\s*(.*)", text)
        if m:
            genres = {g.strip() for g in m.group(1).split(",") if g.strip()}
        else:
            genres = set()
        item2genres[iid] = genres
    return item2genres


def compute_intra_list_similarity(trial_lists: list[list[int]], item_embs: np.ndarray) -> float:
    """推薦リスト内 (Top-K) のアイテム間平均コサイン類似度 (ILS) を計算。"""
    sims = []
    for lst in trial_lists:
        if len(lst) < 2:
            continue
        embs = item_embs[lst]  # (K, dim), L2 normalized
        S = embs @ embs.T     # (K, K)
        # 上三角成分の平均
        k = len(lst)
        triu_idx = np.triu_indices(k, k=1)
        sims.append(float(S[triu_idx].mean()))
    return float(np.mean(sims)) if sims else 0.0


def compute_hit_gt_spread(all_hit_items: list[int], item_embs: np.ndarray) -> float:
    """
    全試行でヒットした Ground Truth アイテム同士の非類似度 (1 - cos_sim) の平均。
    ヒット数が 2 未満の場合は 0.0。
    """
    unique_hits = list(set(all_hit_items))
    if len(unique_hits) < 2:
        return 0.0
    embs = item_embs[unique_hits]  # (N_hits, dim)
    S = embs @ embs.T
    k = len(unique_hits)
    triu_idx = np.triu_indices(k, k=1)
    dists = 1.0 - S[triu_idx]
    return float(dists.mean())


def compute_hit_genre_coverage(all_hit_items: list[int], item2genres: dict[int, set[str]], iid2idx: dict[int, int]) -> int:
    """全試行でヒットした Ground Truth アイテムが属するユニークジャンル数。"""
    idx2iid = {v: k for k, v in iid2idx.items()}
    hit_genres = set()
    for item_idx in all_hit_items:
        iid = idx2iid.get(item_idx)
        if iid in item2genres:
            hit_genres.update(item2genres[iid])
    return len(hit_genres)


def run_verification_eval(
    model, test_gt, train_pos, user_embs, item_embs, item2genres, iid2idx, device
):
    model.prepare(train_pos, user_embs, item_embs, device=device)
    index = model.build_index()

    # 原本の mE5 埋め込み (L2 正規化済み) を距離計算に使用
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
                # 当たった正解アイテム
                hits = [it for it in recs if it in gt]
                all_hit_items.extend(hits)

            per_user["recall_cum"].append(recall_at_k(trial_sets, gt))
            per_user["recall_avg"].append(
                float(np.mean([recall_at_k_single(s, gt) for s in trial_sets]))
            )
            per_user["hit"].append(
                float(np.mean([hit_at_k(s, gt) for s in trial_sets]))
            )
            per_user["temporal_overlap"].append(temporal_overlap_rate(trial_sets, K))
            per_user["ils"].append(compute_intra_list_similarity(trial_lists, norm_item_embs))
            per_user["hgts"].append(compute_hit_gt_spread(all_hit_items, norm_item_embs))
            per_user["hgc"].append(compute_hit_genre_coverage(all_hit_items, item2genres, iid2idx))

        mean_r = {k2: float(np.mean(v)) for k2, v in per_user.items()}
        log.info(
            f"  seed={seed}  rc={mean_r['recall_cum']:.4f}  "
            f"ra={mean_r['recall_avg']:.4f}  "
            f"ILS={mean_r['ils']:.4f}  "
            f"HGTS={mean_r['hgts']:.4f}  "
            f"HGC={mean_r['hgc']:.2f}"
        )
        seed_results.append(mean_r)

    keys = seed_results[0].keys()
    mean = {k2: float(np.mean([r[k2] for r in seed_results])) for k2 in keys}
    std  = {k2: float(np.std( [r[k2] for r in seed_results])) for k2 in keys}
    return {"mean": mean, "std": std}


def plot_verification_results(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0f1117")
    fig.suptitle("Plan 013: Hypothesis Verification for Item Partition Superiority",
                 color="#ffffff", fontsize=16, fontweight="bold", y=0.98)

    models_order = [
        "TwoTower_d2_h64",
        "TwoTower_d2_h64_postnoise_s0p2",
        "TT_divloss_soft_jaccard_l0p1_s0p05",
        "TT_item_partition_n10",
    ]
    labels = ["Base (no-div)", "PostNoise (σ=0.2)", "soft_jaccard (λ=0.1)", "Item Partition (n=10)"]
    colors = ["#555555", "#e74c3c", "#f1c40f", "#2ecc71"]

    metrics_config = [
        ("ils", "Intra-List Similarity (ILS) ↓", "1. List-internal Homogeneity (ILS)\n[Lower = Less Crowded/Occupied]"),
        ("hgts", "Hit GT Spread (HGTS) ↑", "2. Hit Ground-Truth Distance (HGTS)\n[Higher = Hits Multi-Clusters]"),
        ("hgc", "Hit Genre Coverage (HGC) ↑", "3. Hit Movie Genres Count (HGC)\n[Higher = Broader Genre Hit]"),
        ("recall_cum", "Cumulative Recall@10 (rc) ↑", "4. Overall N-Trial Cumulative Recall\n[Higher = Better Search Coverage]"),
    ]

    for ax_idx, (metric_key, ylabel, title) in enumerate(metrics_config):
        ax = axes[ax_idx // 2, ax_idx % 2]
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)

        vals = [all_results[m]["mean"][metric_key] for m in models_order if m in all_results]
        stds = [all_results[m]["std"][metric_key] for m in models_order if m in all_results]
        bars = ax.bar(labels[:len(vals)], vals, color=colors[:len(vals)], alpha=0.85, edgecolor="white", linewidth=0.8)

        # 数値の注記
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.4f}" if metric_key != "hgc" else f"{h:.2f}",
                        (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=10, color="#ffffff", fontweight="bold",
                        xytext=(0, 3), textcoords="offset points")

        ax.set_ylabel(ylabel, color="#e0e0e0", fontsize=11)
        ax.set_title(title, color="#ffffff", fontsize=12, pad=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "hypothesis_verification_013.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info(f"Saved hypothesis plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    log.info("Loading MovieLens 1M data & genres ...")
    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(DATA_DIR)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt   = get_test_gt(interactions, uid2idx, iid2idx)
    item2genres = parse_item_genres(DATA_DIR / "item_texts.parquet")
    log.info(f"Loaded genres for {len(item2genres)} items.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Base TwoTower
    base_tt = TwoTowerModel(
        hidden_dim=BEST_HIDDEN_DIM, depth=BEST_DEPTH,
        lr=1e-3, epochs=50, batch_size=1024,
        logit_scale=14.3, alpha=0.1,
    )
    log.info("Re-training base TwoTower ...")
    base_tt.prepare(train_pos, user_embs, item_embs, device=args.device)

    # 2. PostNoise
    pn = TwoTowerPostNoise(base_tt=base_tt, sigma=0.20)

    # 3. soft_jaccard
    sj = TwoTowerDivLoss(
        base_tt=base_tt, div_loss_name="soft_jaccard",
        lambda_div=0.1, sigma=0.05, lr=2e-3, epochs=30, batch_size=512,
    )

    # 4. Item Partition
    partition = TwoTowerItemPartition(base_tt=base_tt, n_trials=N_TRIALS)

    models = [base_tt, pn, sj, partition]
    all_results = {}

    for model in models:
        log.info(f"\n{'='*60}\nEvaluating Model: {model.name}\n{'='*60}")
        all_results[model.name] = run_verification_eval(
            model, test_gt, train_pos, user_embs, item_embs, item2genres, iid2idx, args.device
        )

    # 保存
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV 出力
    rows = []
    for name, r in all_results.items():
        m = r["mean"]
        rows.append({
            "model": name,
            "recall_cum": m["recall_cum"],
            "recall_avg": m["recall_avg"],
            "hit": m["hit"],
            "ils_intra_list_similarity": m["ils"],
            "hgts_hit_gt_spread": m["hgts"],
            "hgc_hit_genre_coverage": m["hgc"],
        })
    df_res = pd.DataFrame(rows)
    df_res.to_csv(OUT_DIR / "hypothesis_metrics.csv", index=False)
    log.info(f"Saved CSV -> {OUT_DIR / 'hypothesis_metrics.csv'}")

    plot_verification_results(all_results, OUT_DIR)

    log.info("\n" + "="*70 + "\nVerification Summary\n" + "="*70)
    for row in rows:
        log.info(
            f"{row['model']:<38} | rc={row['recall_cum']:.4f} | ra={row['recall_avg']:.4f} | "
            f"ILS(低)= {row['ils_intra_list_similarity']:.4f} | "
            f"HGTS(高)= {row['hgts_hit_gt_spread']:.4f} | "
            f"HGC(高)= {row['hgc_hit_genre_coverage']:.2f}"
        )
    log.info("\n✅ Plan 013 hypothesis verification completed!")


if __name__ == "__main__":
    main()
