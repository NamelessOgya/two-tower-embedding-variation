"""
Plan 002 Experiment Runner
--------------------------
3つのサブ実験を順に実行する:
  2A: M4 Gaussian σ スイープ
  2B: M6 VAE 改良版 4 バリアント
  2C: M5 Dropout 改良版（rate sweep + structured + soft）

Usage:
  python src/run_experiment_002.py --subexp all
  python src/run_experiment_002.py --subexp 2a
  python src/run_experiment_002.py --subexp 2b
  python src/run_experiment_002.py --subexp 2c
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
    load_data, build_index, get_train_pos, get_test_gt,
    evaluate_one_seed, SEEDS, K, N_TRIALS,
)
from src.model.models_002 import (
    make_gaussian_sweep_models, SIGMA_VALUES,
    make_vae_improved_models,
    make_dropout_sweep_models, DROPOUT_RATES,
    M5_StructuredDropout, M5_SoftDropout,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── 共通評価ループ ────────────────────────────────────────────────

def run_models(
    models: list,
    test_gt: dict,
    index: faiss.IndexFlatIP,
    item_embs: np.ndarray,
    train_pos: dict,
    user_embs: np.ndarray,
    device: str,
    n_total_items: int,
) -> dict:
    results = {}
    for model in models:
        log.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        t0 = time.time()
        model.prepare(train_pos, user_embs, item_embs, device=device)
        log.info(f"  prepare: {time.time()-t0:.1f}s")

        seed_results = []
        for seed in SEEDS:
            t0 = time.time()
            m = evaluate_one_seed(
                model, test_gt, index, item_embs,
                k=K, n_trials=N_TRIALS, seed=seed,
                n_total_items=n_total_items,
            )
            log.info(
                f"  seed={seed}  recall_cum={m['recall_cum']:.4f}  "
                f"recall_avg={m['recall_avg']:.4f}  overlap={m['temporal_overlap']:.4f}  "
                f"ild={m['ild']:.4f}  cov={m['coverage']:.4f}  [{time.time()-t0:.1f}s]"
            )
            seed_results.append(m)

        avg = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
        std = {k: float(np.std([r[k] for r in seed_results]))  for k in seed_results[0]}
        log.info(
            f"  [AVG] recall_cum={avg['recall_cum']:.4f}±{std['recall_cum']:.4f}  "
            f"overlap={avg['temporal_overlap']:.4f}±{std['temporal_overlap']:.4f}"
        )
        results[model.name] = {"mean": avg, "std": std, "per_seed": seed_results}
    return results


def save_results(results: dict, out_dir: Path) -> None:
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
    df.to_csv(out_dir / "summary.csv", index=False)
    log.info(f"\n{df.to_string(index=False)}")


# ── 2A: Gaussian σ スイープ ───────────────────────────────────────

def plot_sigma_sweep(results: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0")
        ax.spines[:].set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5)

    sigmas = SIGMA_VALUES
    recall_cum   = [results[f"M4_gauss_sigma{s:.4f}".replace('.','p')]["mean"]["recall_cum"]      for s in sigmas]
    recall_avg= [results[f"M4_gauss_sigma{s:.4f}".replace('.','p')]["mean"]["recall_avg"]    for s in sigmas]
    overlap      = [results[f"M4_gauss_sigma{s:.4f}".replace('.','p')]["mean"]["temporal_overlap"] for s in sigmas]
    diversity    = [1 - o for o in overlap]
    recall_cum_std = [results[f"M4_gauss_sigma{s:.4f}".replace('.','p')]["std"]["recall_cum"]     for s in sigmas]

    cmap = plt.cm.plasma(np.linspace(0.2, 0.9, len(sigmas)))

    # 左: σ vs recall_cum / recall_avg
    ax = axes[0]
    ax.plot(sigmas, recall_cum,    "o-", color="#f4d03f", lw=2, label="recall_cum (cumulative)")
    ax.fill_between(sigmas,
                    [r-e for r,e in zip(recall_cum, recall_cum_std)],
                    [r+e for r,e in zip(recall_cum, recall_cum_std)],
                    color="#f4d03f", alpha=0.2)
    ax.plot(sigmas, recall_avg, "s--", color="#5dade2", lw=2, label="recall_avg (per trial)")
    ax.set_xscale("log")
    ax.set_xlabel("σ (log scale)", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("Recall@10", color="#e0e0e0", fontsize=11)
    ax.set_title("M4 Gaussian: σ vs Recall", color="#e0e0e0", fontsize=12)
    ax.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0")

    # 右: recall_cum vs diversity (フロンティア曲線)
    ax = axes[1]
    sc = ax.scatter(diversity, recall_cum, c=np.log10(sigmas), cmap="plasma",
                    s=120, zorder=3, edgecolors="white", linewidths=0.5)
    ax.plot(diversity, recall_cum, "--", color="#888", lw=1, alpha=0.5)
    for i, s in enumerate(sigmas):
        ax.annotate(f"σ={s}", (diversity[i], recall_cum[i]),
                    textcoords="offset points", xytext=(5, 3),
                    color="#e0e0e0", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log₁₀(σ)", color="#e0e0e0")
    cbar.ax.yaxis.set_tick_params(color="#e0e0e0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e0e0e0")
    ax.set_xlabel("Temporal Diversity (1-Overlap) ↑", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("Recall@10 Cumulative ↑", color="#e0e0e0", fontsize=11)
    ax.set_title("M4 Gaussian: Precision-Diversity Frontier", color="#e0e0e0", fontsize=12)

    fig.suptitle("Sub-exp 2A: Gaussian Noise σ Sweep — MovieLens 1M",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "sigma_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved → {path}")


# ── 統合トレードオフ図 ────────────────────────────────────────────

def plot_combined_tradeoff(all_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#e0e0e0")
    ax.spines[:].set_color("#2d3142")
    ax.grid(True, color="#2d3142", linewidth=0.5)

    # カラーグループ
    groups = {
        "gaussian": ("#f4d03f", "M4 Gaussian"),
        "vae":      ("#e74c3c", "M6 VAE improved"),
        "dropout":  ("#2ecc71", "M5 Dropout improved"),
    }

    def classify(name: str) -> str:
        if "gauss" in name: return "gaussian"
        if "M6" in name:    return "vae"
        return "dropout"

    for name, res in all_results.items():
        diversity = 1 - res["mean"]["temporal_overlap"]
        recall    = res["mean"]["recall_cum"]
        grp = classify(name)
        color, _ = list(groups.values())[list(groups.keys()).index(grp)]
        ax.scatter(diversity, recall, color=color, s=80, alpha=0.8, zorder=3)
        ax.annotate(name.replace("M4_gauss_sigma", "σ=").replace("p", "."),
                    (diversity, recall), textcoords="offset points",
                    xytext=(4, 2), color="#ccc", fontsize=7)

    for grp, (color, label) in groups.items():
        ax.scatter([], [], color=color, s=80, label=label)

    ax.set_xlabel("Temporal Diversity (1-Overlap) ↑", color="#e0e0e0", fontsize=12)
    ax.set_ylabel("Recall@10 Cumulative ↑", color="#e0e0e0", fontsize=12)
    ax.set_title("Plan 002: All Models — Precision-Diversity Tradeoff",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=9)
    plt.tight_layout()
    path = out_dir / "tradeoff_002.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────

def main(subexp: str, device: str) -> None:
    processed_dir = Path("data/processed/movielens")
    report_dir    = Path("report/plan_002")
    report_dir.mkdir(parents=True, exist_ok=True)

    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(processed_dir)
    index     = build_index(item_embs)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt   = get_test_gt(interactions, uid2idx, iid2idx)

    all_results: dict = {}

    if subexp in ("2a", "all"):
        log.info("\n\n" + "="*70 + "\nSub-exp 2A: Gaussian σ sweep\n" + "="*70)
        models = make_gaussian_sweep_models()
        res = run_models(models, test_gt, index, item_embs, train_pos,
                         user_embs, device, len(item_embs))
        save_results(res, report_dir / "sigma_sweep")
        plot_sigma_sweep(res, report_dir / "sigma_sweep")
        all_results.update(res)

    if subexp in ("2b", "all"):
        log.info("\n\n" + "="*70 + "\nSub-exp 2B: VAE improved\n" + "="*70)
        models = make_vae_improved_models()
        res = run_models(models, test_gt, index, item_embs, train_pos,
                         user_embs, device, len(item_embs))
        save_results(res, report_dir / "vae_improved")
        all_results.update(res)

    if subexp in ("2c", "all"):
        log.info("\n\n" + "="*70 + "\nSub-exp 2C: Dropout improved\n" + "="*70)
        models = (
            make_dropout_sweep_models()
            + [M5_StructuredDropout(n_groups=12, group_dropout_rate=0.2)]
            + [M5_SoftDropout(noise_rate=0.2, sigma=0.02),
               M5_SoftDropout(noise_rate=0.3, sigma=0.05)]
        )
        res = run_models(models, test_gt, index, item_embs, train_pos,
                         user_embs, device, len(item_embs))
        save_results(res, report_dir / "dropout_improved")
        all_results.update(res)

    if all_results:
        plot_combined_tradeoff(all_results, report_dir)
        with open(report_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"\nAll results saved → {report_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all", choices=["all", "2a", "2b", "2c"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.subexp, args.device)
