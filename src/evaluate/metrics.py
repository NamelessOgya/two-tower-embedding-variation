"""Evaluation metrics for recommendation diversity experiments."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def recall_cum(trial_sets: list[set], ground_truth: set) -> float:
    """recall_cum: N試行の推薦リストの和集合に対する recall.

    recall_cum = |union(R_1, ..., R_N) ∩ GT| / |GT|

    多様化により何回かの試行のうちどれかで正解を見つけた割合。
    N試行の結果を蓄積するため、多様性が高いほど上昇する。
    """
    if not ground_truth:
        return 0.0
    union = set().union(*trial_sets) if trial_sets else set()
    return len(union & ground_truth) / len(ground_truth)

# 後方互換エイリアス
recall_at_k = recall_cum


def recall_avg(trial_sets: list[set], ground_truth: set) -> float:
    """recall_avg: 1試行あたりの recall の平均.

    recall_avg = mean_t [ |R_t ∩ GT| / |GT| ]

    各試行の精度（推薦精度）を平均したもの。
    多様性の影響を受けず、1回あたりの推薦の質を表す。
    """
    if not ground_truth:
        return 0.0
    return float(np.mean([recall_at_k_single(s, ground_truth) for s in trial_sets]))


def recall_at_k_single(recommended: set, ground_truth: set) -> float:
    """1試行・1ユーザーの recall (内部ヘルパー)."""
    if not ground_truth:
        return 0.0
    return len(recommended & ground_truth) / len(ground_truth)


def hit_at_k(recommended: set, ground_truth: set) -> float:
    """1 if any GT item appears in the recommendation list."""
    return float(len(recommended & ground_truth) > 0)


def ndcg_at_k(recommended_list: list[int], ground_truth: set, k: int) -> float:
    """NDCG@K for a single ranked recommendation list."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended_list[:k])
        if item in ground_truth
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / ideal if ideal > 0 else 0.0


def temporal_overlap_rate(trial_sets: list[set], k: int) -> float:
    """Average pairwise Jaccard overlap (normalised by K) across all trial pairs.

    overlap(t1, t2) = |R(t1) ∩ R(t2)| / K
    """
    if len(trial_sets) < 2:
        return 1.0
    rates = [len(s1 & s2) / k for s1, s2 in combinations(trial_sets, 2)]
    return float(np.mean(rates))


def intra_list_diversity(
    recommended_list: list[int],
    item_embeddings: np.ndarray,
    k: int,
) -> float:
    """Average pairwise cosine distance within a recommendation list.

    Since item_embeddings are L2-normalised, cosine distance = 1 − dot(vi, vj).
    """
    items = recommended_list[:k]
    if len(items) < 2:
        return 0.0
    embs = item_embeddings[items]  # (≤k, 768), already L2-normalised
    sim = embs @ embs.T            # cosine similarity matrix
    n = len(items)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return 0.0
    dists = [1.0 - sim[i, j] for i, j in pairs]
    return float(np.mean(dists))


def coverage(all_recommended: set, total_items: int) -> float:
    """Fraction of the item catalogue ever recommended."""
    return len(all_recommended) / total_items if total_items > 0 else 0.0
