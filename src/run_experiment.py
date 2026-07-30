"""
Main Experiment Runner – plan_001
----------------------------------
Evaluates M0–M6 on MovieLens 1M (and optionally Yelp).

Pipeline:
  1. Load preprocessed interactions + embeddings
  2. Build FAISS CPU index on item embeddings
  3. For each model:
       a. prepare() – fit any model-specific parameters
       b. For each of 5 seeds:
            For each test user:
              Run N_TRIALS recommendation trials
              Compute per-user metrics
       c. Average metrics over seeds
  4. Save result/plan_001/{dataset}/results.json + summary.csv

Usage:
  python src/run_experiment.py --dataset movielens
  python src/run_experiment.py --dataset yelp
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.model.models import (
    M0_Baseline,
    M1_Clustering,
    M2_RandomAttention,
    M3_RandomSubset,
    M4_GaussianNoise,
    M5_MCDropout,
    M6_VAE,
)
from src.evaluate.metrics import (
    recall_at_k,
    recall_at_k_single,
    hit_at_k,
    ndcg_at_k,
    temporal_overlap_rate,
    intra_list_diversity,
    coverage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SEEDS = [0, 1, 2, 3, 4]
K = 10
N_TRIALS = 10


# ── Data Helpers ──────────────────────────────────────────────────────────────

def load_data(processed_dir: Path):
    log.info(f"Loading data from {processed_dir}")
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    user_embs = np.load(processed_dir / "user_embeddings.npy").astype(np.float32)
    item_embs = np.load(processed_dir / "item_embeddings.npy").astype(np.float32)
    user_id_map = pd.read_parquet(processed_dir / "user_id_map.parquet")
    item_id_map = pd.read_parquet(processed_dir / "item_id_map.parquet")

    uid2idx = dict(zip(user_id_map["user_id"], user_id_map["index"]))
    iid2idx = dict(zip(item_id_map["item_id"], item_id_map["index"]))

    log.info(f"  Users={len(user_embs)}  Items={len(item_embs)}  "
             f"Interactions={len(interactions)}")
    return interactions, user_embs, item_embs, uid2idx, iid2idx


def build_index(item_embs: np.ndarray) -> faiss.IndexFlatIP:
    log.info("Building FAISS IndexFlatIP (CPU) ...")
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)
    log.info(f"  Index size: {index.ntotal}")
    return index


def get_train_pos(interactions, uid2idx, iid2idx) -> dict[int, np.ndarray]:
    train = interactions[
        (interactions["split"] == "train") & (interactions["is_positive"])
    ]
    result: dict[int, np.ndarray] = {}
    for uid, grp in train.groupby("user_id"):
        uidx = uid2idx.get(uid)
        if uidx is None:
            continue
        idxs = [iid2idx[iid] for iid in grp["item_id"] if iid in iid2idx]
        if idxs:
            result[uidx] = np.array(idxs, dtype=np.int64)
    return result


def get_test_gt(interactions, uid2idx, iid2idx) -> dict[int, set]:
    test = interactions[
        (interactions["split"] == "test") & (interactions["is_positive"])
    ]
    result: dict[int, set] = {}
    for uid, grp in test.groupby("user_id"):
        uidx = uid2idx.get(uid)
        if uidx is None:
            continue
        idxs = {iid2idx[iid] for iid in grp["item_id"] if iid in iid2idx}
        if idxs:
            result[uidx] = idxs
    return result


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_one_seed(
    model,
    test_gt: dict[int, set],
    index: faiss.IndexFlatIP,
    item_embs: np.ndarray,
    k: int,
    n_trials: int,
    seed: int,
    n_total_items: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    per_user: dict[str, list[float]] = defaultdict(list)
    global_recommended: set[int] = set()

    for user_idx, gt in test_gt.items():
        trial_lists: list[list[int]] = []
        trial_sets: list[set[int]] = []

        for trial in range(n_trials):
            q = model.get_query_vector(user_idx, trial, rng).reshape(1, -1)
            _, I = index.search(q, k)
            recs = list(map(int, I[0]))
            trial_lists.append(recs)
            trial_sets.append(set(recs))
            global_recommended.update(recs)

        per_user["recall_cum"].append(recall_at_k(trial_sets, gt))
        per_user["recall_avg"].append(
            float(np.mean([recall_at_k_single(s, gt) for s in trial_sets]))
        )
        per_user["hit"].append(
            float(np.mean([hit_at_k(s, gt) for s in trial_sets]))
        )
        per_user["ndcg"].append(
            float(np.mean([ndcg_at_k(lst, gt, k) for lst in trial_lists]))
        )
        per_user["temporal_overlap"].append(temporal_overlap_rate(trial_sets, k))
        per_user["ild"].append(
            float(np.mean([intra_list_diversity(lst, item_embs, k) for lst in trial_lists]))
        )

    results = {key: float(np.mean(vals)) for key, vals in per_user.items()}
    results["coverage"] = coverage(global_recommended, n_total_items)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment(
    dataset: str,
    result_dir: Path,
    device: str = "cuda",
) -> None:
    processed_dir = Path(f"data/processed/{dataset}")
    out_dir = result_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(processed_dir)
    index = build_index(item_embs)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt = get_test_gt(interactions, uid2idx, iid2idx)
    log.info(f"Test users with ground truth: {len(test_gt)}")

    # ── Build model list ──────────────────────────────────────────────────────
    variants_path = processed_dir / "user_embeddings_variants.npy"
    models = [
        M0_Baseline(),
        M1_Clustering(n_clusters=5),
        M2_RandomAttention(),
        M4_GaussianNoise(sigma=0.05),
        M5_MCDropout(dropout_rate=0.2),
        M6_VAE(latent_dim=128, beta=1.0, epochs=100, batch_size=256),
    ]
    if variants_path.exists():
        models.insert(3, M3_RandomSubset(str(variants_path)))
        log.info("M3 variants found – including M3 in experiment.")
    else:
        log.warning(f"M3 variants not found at {variants_path} – skipping M3.")

    all_results: dict = {}

    for model in models:
        sep = "=" * 65
        log.info(f"\n{sep}\nModel: {model.name}\n{sep}")

        t0 = time.time()
        model.prepare(train_pos, user_embs, item_embs, device=device)
        log.info(f"  prepare: {time.time()-t0:.1f}s")

        seed_results: list[dict] = []
        for seed in SEEDS:
            t0 = time.time()
            m = evaluate_one_seed(
                model, test_gt, index, item_embs,
                k=K, n_trials=N_TRIALS, seed=seed,
                n_total_items=len(item_embs),
            )
            elapsed = time.time() - t0
            log.info(
                f"  seed={seed}  "
                f"recall_cum={m['recall_cum']:.4f}  "
                f"recall_avg={m['recall_avg']:.4f}  "
                f"hit={m['hit']:.4f}  "
                f"ndcg={m['ndcg']:.4f}  "
                f"overlap={m['temporal_overlap']:.4f}  "
                f"ild={m['ild']:.4f}  "
                f"cov={m['coverage']:.4f}  "
                f"[{elapsed:.1f}s]"
            )
            seed_results.append(m)

        # Average over seeds
        avg, std = {}, {}
        for key in seed_results[0]:
            vals = [r[key] for r in seed_results]
            avg[key] = float(np.mean(vals))
            std[key] = float(np.std(vals))

        log.info(
            f"  [AVG]  "
            f"recall_cum={avg['recall_cum']:.4f}±{std['recall_cum']:.4f}  "
            f"recall_avg={avg['recall_avg']:.4f}±{std['recall_avg']:.4f}  "
            f"overlap={avg['temporal_overlap']:.4f}±{std['temporal_overlap']:.4f}  "
            f"ild={avg['ild']:.4f}±{std['ild']:.4f}  "
            f"cov={avg['coverage']:.4f}±{std['coverage']:.4f}"
        )

        all_results[model.name] = {
            "mean": avg,
            "std": std,
            "per_seed": seed_results,
        }

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved full results → {results_path}")

    # Summary table
    metric_keys = list(next(iter(all_results.values()))["mean"].keys())
    rows = []
    for model_name, res in all_results.items():
        row = {"model": model_name}
        for k_name in metric_keys:
            row[k_name] = f"{res['mean'][k_name]:.4f}±{res['std'][k_name]:.4f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Saved summary → {summary_path}")

    # Pretty print
    log.info("\n" + "=" * 120)
    log.info("EXPERIMENT SUMMARY")
    log.info("=" * 120)
    log.info("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run plan_001 experiment")
    parser.add_argument(
        "--dataset", type=str, default="movielens",
        choices=["movielens", "yelp"],
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("result/plan_001"),
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="PyTorch device for M6 VAE training (default: cuda)",
    )
    args = parser.parse_args()
    run_experiment(args.dataset, args.result_dir, args.device)
