"""
Plan 005 Experiment Runner
--------------------------
5A: Ablation Study for Model Enhancements
    - M0_raw: Baseline text embeddings
    - M0_whiten: Embedding Whitening (ZCA/PCA)
    - M0_logq: Log-Q popularity correction (scale=1.0)
    - M0_scaled_logq: Logit Scaling (scale=14.3) + Log-Q
    - M0_strong: Whitening + Logit Scaling + Log-Q

5B: M4 Gaussian Noise Sweep on the BEST 5A Model
    - sigma in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

Usage:
    python src/run_experiment_005.py --subexp all --device cuda
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.run_experiment import (
    load_data, get_train_pos, get_test_gt,
    SEEDS, K, N_TRIALS,
)
from src.evaluate.metrics import (
    recall_at_k, recall_at_k_single, hit_at_k, ndcg_at_k,
    temporal_overlap_rate, intra_list_diversity, coverage,
)
from src.model.models_005 import (
    M0_EnhancedBase, M4_GaussOnEnhancedModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SIGMA_SWEEP = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def build_faiss_index(item_embs: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs.astype(np.float32))
    return index


def evaluate_model_seed(
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
            if hasattr(model, "recommend"):
                recs = model.recommend(user_idx, trial, rng, index, k)
            else:
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


def run_models_eval(models, test_gt, train_pos, user_embs, item_embs, device, n_total_items):
    results = {}
    for model in models:
        log.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        t0 = time.time()
        model.prepare(train_pos, user_embs, item_embs, device=device)
        log.info(f"  prepare: {time.time()-t0:.1f}s")

        # Build index based on model's transformed item embeddings
        curr_item_embs = (
            model.transformed_item_embs
            if hasattr(model, "transformed_item_embs") and model.transformed_item_embs is not None
            else item_embs
        )
        index = build_faiss_index(curr_item_embs)

        seed_results = []
        for seed in SEEDS:
            m = evaluate_model_seed(
                model, test_gt, index, curr_item_embs,
                k=K, n_trials=N_TRIALS, seed=seed,
                n_total_items=n_total_items,
            )
            log.info(
                f"  seed={seed}  rc={m['recall_cum']:.4f}  ra={m['recall_avg']:.4f}  "
                f"ov={m['temporal_overlap']:.4f}  cov={m['coverage']:.4f}"
            )
            seed_results.append(m)

        avg = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
        std = {k: float(np.std([r[k] for r in seed_results]))  for k in seed_results[0]}
        log.info(
            f"  [AVG] rc={avg['recall_cum']:.4f}±{std['recall_cum']:.4f}  "
            f"ra={avg['recall_avg']:.4f}±{std['recall_avg']:.4f}  "
            f"ov={avg['temporal_overlap']:.4f}±{std['temporal_overlap']:.4f}"
        )
        results[model.name] = {"mean": avg, "std": std, "per_seed": seed_results}
    return results


def save_summary(results: dict, out_dir: Path, filename: str = "summary.csv"):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    rows = []
    for name, r in results.items():
        row = {"model": name}
        for k in r["mean"]:
            row[k] = f"{r['mean'][k]:.4f}±{r['std'][k]:.4f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / filename, index=False)
    log.info(f"\n{df.to_string(index=False)}")


def plot_5b_tradeoff(sweep_results: dict, best_5a_name: str, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.7)

    ax0, ax1 = axes

    divs, ras, rcs, sigmas = [], [], [], []
    for name, r in sweep_results.items():
        m = r["mean"]
        div = 1.0 - m["temporal_overlap"]
        ra = m["recall_avg"]
        rc = m["recall_cum"]
        # extract sigma from name
        sig_str = name.split("_s")[-1].replace("p", ".")
        try:
            sig = float(sig_str)
        except ValueError:
            sig = 0.0
        divs.append(div)
        ras.append(ra)
        rcs.append(rc)
        sigmas.append(sig)

    # Sort by diversity
    order = np.argsort(divs)
    divs = [divs[i] for i in order]
    ras = [ras[i] for i in order]
    rcs = [rcs[i] for i in order]
    sigmas = [sigmas[i] for i in order]

    ax0.plot(divs, ras, "o-", color="#e74c3c", lw=2, label=f"M4 Gauss on {best_5a_name}")
    for d, r, s in zip(divs, ras, sigmas):
        ax0.annotate(f"σ={s:.3f}", (d, r), fontsize=7, color="#f39c12", xytext=(3, 3), textcoords="offset points")

    ax0.set_xlabel("Diversity (1 − temporal_overlap) ↑", color="#e0e0e0", fontsize=10)
    ax0.set_ylabel("recall_avg ↑ (1-trial precision)", color="#e0e0e0", fontsize=10)
    ax0.set_title("5B: Precision (recall_avg) vs Diversity", color="#e0e0e0", fontsize=12)
    ax0.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=9)

    ax1.plot(divs, rcs, "s-", color="#2ecc71", lw=2, label=f"M4 Gauss on {best_5a_name}")
    for d, r, s in zip(divs, rcs, sigmas):
        ax1.annotate(f"σ={s:.3f}", (d, r), fontsize=7, color="#f39c12", xytext=(3, 3), textcoords="offset points")

    ax1.set_xlabel("Diversity (1 − temporal_overlap) ↑", color="#e0e0e0", fontsize=10)
    ax1.set_ylabel("recall_cum ↑ (N-trial cumulative)", color="#e0e0e0", fontsize=10)
    ax1.set_title("5B: Cumulative Recall (recall_cum) vs Diversity", color="#e0e0e0", fontsize=12)
    ax1.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=9)

    fig.suptitle(f"Plan 005: Trade-off Curves for M4 Gaussian on Best Model ({best_5a_name})", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "tradeoff_005.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved trade-off plot → {path}")


def main(subexp: str, device: str):
    processed_dir = Path("data/processed/movielens")
    report_dir = Path("report/plan_005")
    report_dir.mkdir(parents=True, exist_ok=True)

    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(processed_dir)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt = get_test_gt(interactions, uid2idx, iid2idx)

    best_5a_model_obj = None
    best_5a_name = "M0_strong"

    # ─── Sub-exp 5A: Ablation Study ──────────────────────────────────────────
    if subexp in ("5a", "all"):
        log.info("\n" + "="*70 + "\nSub-exp 5A: Base Model Enhancement Ablation Study\n" + "="*70)

        ablation_models = [
            M0_EnhancedBase(name="M0_raw", use_whitening=False, use_logq=False),
            M0_EnhancedBase(name="M0_whiten", use_whitening=True, use_logq=False),
            M0_EnhancedBase(name="M0_logq", use_whitening=False, use_logq=True, logit_scale=1.0, alpha=0.1),
            M0_EnhancedBase(name="M0_scaled_logq", use_whitening=False, use_logq=True, logit_scale=14.3, alpha=0.1),
            M0_EnhancedBase(name="M0_strong", use_whitening=True, use_logq=True, logit_scale=14.3, alpha=0.1),
        ]

        out_dir_5a = report_dir / "ablation"
        res_5a = run_models_eval(ablation_models, test_gt, train_pos, user_embs, item_embs, device, len(item_embs))
        save_summary(res_5a, out_dir_5a)

        # Select best 5A model based on highest recall_cum (or recall_avg)
        best_name = max(res_5a.keys(), key=lambda k: res_5a[k]["mean"]["recall_cum"])
        best_rc = res_5a[best_name]["mean"]["recall_cum"]
        best_ra = res_5a[best_name]["mean"]["recall_avg"]
        log.info(f"\n🏆 Best 5A Model Selected: {best_name} (recall_cum={best_rc:.4f}, recall_avg={best_ra:.4f})")

        best_5a_name = best_name
        best_5a_model_obj = next(m for m in ablation_models if m.name == best_name)

    # ─── Sub-exp 5B: M4 Gaussian Sweep on Best 5A Model ─────────────────────
    if subexp in ("5b", "all"):
        log.info("\n" + "="*70 + f"\nSub-exp 5B: M4 Gaussian Noise Sweep on Best Model ({best_5a_name})\n" + "="*70)

        if best_5a_model_obj is None:
            # Fallback if 5B run independently
            best_5a_model_obj = M0_EnhancedBase(name=best_5a_name, use_whitening=True, use_logq=True, logit_scale=14.3, alpha=0.1)
            best_5a_model_obj.prepare(train_pos, user_embs, item_embs, device=device)

        sweep_models = [
            M4_GaussOnEnhancedModel(base_model=best_5a_model_obj, sigma=sig)
            for sig in SIGMA_SWEEP
        ]

        out_dir_5b = report_dir / "gauss_sweep"
        res_5b = run_models_eval(sweep_models, test_gt, train_pos, user_embs, item_embs, device, len(item_embs))
        save_summary(res_5b, out_dir_5b)

        plot_5b_tradeoff(res_5b, best_5a_name, report_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all", choices=["all", "5a", "5b"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.subexp, args.device)
