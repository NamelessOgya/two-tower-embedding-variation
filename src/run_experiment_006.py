"""
Plan 006 Experiment Runner (with Micro-Lambda Sweep for 3B Adapters)
---------------------------------------------------------------------
Evaluates all diversity methods on top of M0_strong (Whitened + CLIP Logit Scaling + Log-Q):
  - 6A: M4_strong Gaussian noise (sigma sweep)
  - 6B: M5_strong MC Dropout (p sweep)
  - 6C: 3B_strong Diversity Adapter (micro-lambda sweep: 0.0001 to 2.0 across 4 losses)
  - 6D: Unified Pareto Frontier Plots

Usage:
    PYTHONPATH=. python3 src/run_experiment_006.py --subexp all --device cuda
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
from src.model.models_005 import M0_EnhancedBase
from src.model.models_006 import (
    M4_strong_Gauss, M5_strong_Dropout, M3B_strong_Adapter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SIGMA_SWEEP = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
DROPOUT_SWEEP = [0.05, 0.10, 0.20, 0.30, 0.50]
LAMBDA_SWEEP = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
SWEEP_LOSSES = ["cosine_emb", "l2_emb", "soft_jaccard", "listnet"]


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


def plot_unified_006(results: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.7)

    ax0, ax1 = axes

    colors = {
        "M0_strong": "#ffffff",
        "M4_gauss":  "#e74c3c",
        "M5_dropout": "#f39c12",
        "cosine_emb": "#2ecc71",
        "l2_emb":     "#3498db",
        "soft_jaccard": "#9b59b6",
        "listnet":    "#1abc9c",
    }

    # 1. M0_strong baseline
    if "M0_strong" in results:
        m0 = results["M0_strong"]["mean"]
        ax0.scatter([0.0], [m0["recall_avg"]], color="white", marker="*", s=250, zorder=10, label="M0_strong (baseline)")
        ax1.scatter([0.0], [m0["recall_cum"]], color="white", marker="*", s=250, zorder=10, label="M0_strong (baseline)")

    # 2. M4 Gaussian Noise
    m4_pts = []
    for name, r in results.items():
        if name.startswith("M4_strong_gauss"):
            m = r["mean"]
            div = 1.0 - m["temporal_overlap"]
            sig_str = name.split("_s")[-1].replace("p", ".")
            try:
                sig = float(sig_str)
            except ValueError:
                sig = 0.0
            m4_pts.append((div, m["recall_avg"], m["recall_cum"], sig))
    m4_pts.sort(key=lambda x: x[0])
    if m4_pts:
        d = [p[0] for p in m4_pts]
        ra = [p[1] for p in m4_pts]
        rc = [p[2] for p in m4_pts]
        ax0.plot(d, ra, "o-", color=colors["M4_gauss"], lw=2, label="M4 Gaussian (σ sweep)")
        ax1.plot(d, rc, "o-", color=colors["M4_gauss"], lw=2, label="M4 Gaussian (σ sweep)")

    # 3. M5 Dropout
    m5_pts = []
    for name, r in results.items():
        if name.startswith("M5_strong_dropout"):
            m = r["mean"]
            div = 1.0 - m["temporal_overlap"]
            m5_pts.append((div, m["recall_avg"], m["recall_cum"]))
    m5_pts.sort(key=lambda x: x[0])
    if m5_pts:
        d = [p[0] for p in m5_pts]
        ra = [p[1] for p in m5_pts]
        rc = [p[2] for p in m5_pts]
        ax0.plot(d, ra, "s--", color=colors["M5_dropout"], lw=2, label="M5 Dropout (p sweep)")
        ax1.plot(d, rc, "s--", color=colors["M5_dropout"], lw=2, label="M5 Dropout (p sweep)")

    # 4. 3B Diversity Adapters
    for loss in SWEEP_LOSSES:
        pts = []
        for name, r in results.items():
            prefix = f"3B_strong_{loss}_l"
            if name.startswith(prefix):
                m = r["mean"]
                div = 1.0 - m["temporal_overlap"]
                lam_str = name[len(prefix):].replace("p", ".")
                try:
                    lam = float(lam_str)
                except ValueError:
                    lam = 0.0
                pts.append((div, m["recall_avg"], m["recall_cum"], lam))
        pts.sort(key=lambda x: x[0])
        if pts:
            d = [p[0] for p in pts]
            ra = [p[1] for p in pts]
            rc = [p[2] for p in pts]
            col = colors.get(loss, "#aaaaaa")
            ax0.plot(d, ra, "D-.", color=col, lw=2, label=f"3B {loss} (λ micro-sweep)")
            ax1.plot(d, rc, "D-.", color=col, lw=2, label=f"3B {loss} (λ micro-sweep)")
            for div_val, rc_val, lam_val in zip(d, rc, [p[3] for p in pts]):
                if lam_val in (0.05, 0.1, 1.0):
                    ax1.annotate(f"λ={lam_val}", (div_val, rc_val), fontsize=7, color=col, xytext=(3, 3), textcoords="offset points")

    ax0.set_xlabel("Diversity (1 − temporal_overlap) ↑", color="#e0e0e0", fontsize=10)
    ax0.set_ylabel("recall_avg ↑ (1-trial precision)", color="#e0e0e0", fontsize=10)
    ax0.set_title("Plan 006: Precision (recall_avg) vs Diversity — Strong Baseline", color="#e0e0e0", fontsize=11)
    ax0.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=8)

    ax1.set_xlabel("Diversity (1 − temporal_overlap) ↑", color="#e0e0e0", fontsize=10)
    ax1.set_ylabel("recall_cum ↑ (N-trial cumulative)", color="#e0e0e0", fontsize=10)
    ax1.set_title("Plan 006: Cumulative Recall (recall_cum) vs Diversity — Strong Baseline", color="#e0e0e0", fontsize=11)
    ax1.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=8)

    fig.suptitle("Plan 006: Unified Tradeoff Frontier (Micro-λ Sweep) on M0_strong Baseline", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "tradeoff_006_unified.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved unified tradeoff plot → {path}")


def main(subexp: str, device: str):
    processed_dir = Path("data/processed/movielens")
    report_dir = Path("report/plan_006")
    report_dir.mkdir(parents=True, exist_ok=True)

    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(processed_dir)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt = get_test_gt(interactions, uid2idx, iid2idx)

    all_results = {}
    res_json = report_dir / "results.json"
    if res_json.exists():
        try:
            with open(res_json) as f:
                all_results = json.load(f)
            log.info(f"Loaded {len(all_results)} existing model results from {res_json}")
        except Exception as e:
            log.warning(f"Could not load {res_json}: {e}")

    # M0_strong baseline reference
    if "M0_strong" not in all_results:
        m0_strong = M0_EnhancedBase(name="M0_strong", use_whitening=True, use_logq=True, logit_scale=14.3, alpha=0.1)
        res_m0 = run_models_eval([m0_strong], test_gt, train_pos, user_embs, item_embs, device, len(item_embs))
        all_results.update(res_m0)

    # ─── 6A: M4 Gaussian Noise ───────────────────────────────────────────────
    if subexp in ("6a", "all"):
        log.info("\n" + "="*70 + "\nSub-exp 6A: M4 Gaussian Noise Sweep on M0_strong\n" + "="*70)
        models_6a = [M4_strong_Gauss(sigma=sig) for sig in SIGMA_SWEEP]
        res_6a = run_models_eval(models_6a, test_gt, train_pos, user_embs, item_embs, device, len(item_embs))
        all_results.update(res_6a)

    # ─── 6B: M5 MC Dropout ──────────────────────────────────────────────────
    if subexp in ("6b", "all"):
        log.info("\n" + "="*70 + "\nSub-exp 6B: M5 MC Dropout Sweep on M0_strong\n" + "="*70)
        models_6b = [M5_strong_Dropout(dropout_rate=p) for p in DROPOUT_SWEEP]
        res_6b = run_models_eval(models_6b, test_gt, train_pos, user_embs, item_embs, device, len(item_embs))
        all_results.update(res_6b)

    # ─── 6C: 3B Diversity Adapter Micro-Lambda Sweep ────────────────────────
    if subexp in ("6c", "all"):
        log.info("\n" + "="*70 + "\nSub-exp 6C: 3B Diversity Adapter Micro-Lambda Sweep on M0_strong\n" + "="*70)
        models_6c = [
            M3B_strong_Adapter(div_loss_name=loss, lambda_div=lam)
            for loss in SWEEP_LOSSES
            for lam in LAMBDA_SWEEP
        ]
        res_6c = run_models_eval(models_6c, test_gt, train_pos, user_embs, item_embs, device, len(item_embs))
        all_results.update(res_6c)

    # Save summary and unified plot
    save_summary(all_results, report_dir)
    plot_unified_006(all_results, report_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all", choices=["all", "6a", "6b", "6c"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.subexp, args.device)
