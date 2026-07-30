"""
Plan 003 Experiment Runner
--------------------------
3A: Gaussian ノイズ挿入位置の比較
    positions = [input, middle, output]
    sigmas    = [0.01, 0.02, 0.05]

3B: 多様性損失付き学習アダプタ
    loss_fns  = [cosine_emb, l2_emb, kl_dist, js_dist, soft_jaccard, listnet]
    lambda    = 1.0 (固定; チューニングは今後)

Usage:
  python src/run_experiment_003.py --subexp all
  python src/run_experiment_003.py --subexp 3a
  python src/run_experiment_003.py --subexp 3b
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.run_experiment import (
    load_data, build_index, get_train_pos, get_test_gt,
    evaluate_one_seed, SEEDS, K, N_TRIALS,
)
from src.model.models_003 import (
    precompute_all_noisy_embeddings,
    M_NoisePosition, POSITIONS_3A, SIGMAS_3A,
    make_diversity_adapter_models, DIV_LOSSES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── 共通評価ループ ─────────────────────────────────────────────────

def run_models(models, test_gt, index, item_embs, train_pos, user_embs, device, n_total_items):
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


def save_results(results, out_dir: Path):
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


# ── 3A: ノイズ位置比較プロット ─────────────────────────────────────

def plot_noise_position(results: dict, out_dir: Path):
    """
    1枚目: sigma vs recall_cum (位置ごとに色分け)
    2枚目: diversity vs recall_cum フロンティア曲線
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0")
        ax.spines[:].set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5)

    pos_colors = {"input": "#e74c3c", "middle": "#f39c12", "output": "#2ecc71"}
    pos_markers = {"input": "o", "middle": "s", "output": "^"}
    pos_labels  = {"input": "Input (token emb.)", "middle": "Middle (layer 6/12)", "output": "Output (post-pool)"}

    ax0, ax1 = axes

    for pos in POSITIONS_3A:
        sigmas = SIGMAS_3A
        names = [f"3A_{pos}_s{s:.3f}".replace(".", "p") for s in sigmas]
        rc = [results[n]["mean"]["recall_cum"]      if n in results else np.nan for n in names]
        ov = [results[n]["mean"]["temporal_overlap"] if n in results else np.nan for n in names]
        rc_std = [results[n]["std"]["recall_cum"]   if n in results else 0 for n in names]
        div = [1 - o for o in ov]

        c = pos_colors[pos]; mk = pos_markers[pos]; lbl = pos_labels[pos]
        ax0.plot(sigmas, rc, marker=mk, color=c, lw=2, label=lbl)
        ax0.fill_between(sigmas, [r-e for r, e in zip(rc, rc_std)],
                         [r+e for r, e in zip(rc, rc_std)], color=c, alpha=0.15)
        for i, s in enumerate(sigmas):
            ax1.scatter(div[i], rc[i], color=c, marker=mk, s=120,
                        edgecolors="white", linewidths=0.5, zorder=3)
            ax1.annotate(f"σ={s}", (div[i], rc[i]),
                         textcoords="offset points", xytext=(4, 3),
                         color="#ccc", fontsize=7)

    ax0.set_xscale("log")
    ax0.set_xlabel("σ (log scale)", color="#e0e0e0", fontsize=11)
    ax0.set_ylabel("Recall@10 Cumulative ↑", color="#e0e0e0", fontsize=11)
    ax0.set_title("3A: Noise Position × σ → Recall", color="#e0e0e0", fontsize=12)
    ax0.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=9)

    for pos in POSITIONS_3A:
        ax1.scatter([], [], color=pos_colors[pos], marker=pos_markers[pos], s=80, label=pos_labels[pos])
    ax1.set_xlabel("Temporal Diversity (1 - Overlap) ↑", color="#e0e0e0", fontsize=11)
    ax1.set_ylabel("Recall@10 Cumulative ↑", color="#e0e0e0", fontsize=11)
    ax1.set_title("3A: Precision-Diversity Frontier by Position", color="#e0e0e0", fontsize=12)
    ax1.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=9)

    fig.suptitle("Sub-exp 3A: Gaussian Noise Insertion Position — MovieLens 1M",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "noise_position.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved → {path}")


# ── 3B: 多様性アダプタプロット ──────────────────────────────────────

DIV_COLORS = {
    "cosine_emb":   "#f4d03f",
    "l2_emb":       "#e67e22",
    "kl_dist":      "#3498db",
    "js_dist":      "#9b59b6",
    "soft_jaccard": "#e74c3c",
    "listnet":      "#1abc9c",
}
DIV_LABELS = {
    "cosine_emb":   "Cosine (emb)",
    "l2_emb":       "L2 (emb)",
    "kl_dist":      "KL (score dist)",
    "js_dist":      "JS (score dist)",
    "soft_jaccard": "Soft Jaccard (list)",
    "listnet":      "ListNet (rank)",
}


def plot_diversity_adapter(results: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0")
        ax.spines[:].set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5)

    ax0, ax1 = axes

    for loss_name in DIV_LOSSES:
        name = f"3B_{loss_name}_l1p0"
        if name not in results: continue
        r = results[name]
        div = 1 - r["mean"]["temporal_overlap"]
        rc  = r["mean"]["recall_cum"]
        rc1 = r["mean"]["recall_avg"]
        color = DIV_COLORS.get(loss_name, "#888")
        label = DIV_LABELS.get(loss_name, loss_name)

        ax0.scatter(div, rc,  color=color, s=120, edgecolors="white", linewidths=0.5, zorder=3)
        ax0.annotate(label, (div, rc), textcoords="offset points", xytext=(5, 2),
                     color="#ccc", fontsize=7)
        ax1.scatter(div, rc1, color=color, s=120, edgecolors="white", linewidths=0.5, zorder=3,
                    label=label)
        ax1.annotate(label, (div, rc1), textcoords="offset points", xytext=(5, 2),
                     color="#ccc", fontsize=7)

    ax0.set_xlabel("Temporal Diversity (1-Overlap) ↑", color="#e0e0e0", fontsize=11)
    ax0.set_ylabel("recall_cum↑ (multi-trial)", color="#e0e0e0", fontsize=11)
    ax0.set_title("3B: Diversity Loss — recall_cum vs Diversity", color="#e0e0e0", fontsize=12)

    ax1.set_xlabel("Temporal Diversity (1-Overlap) ↑", color="#e0e0e0", fontsize=11)
    ax1.set_ylabel("recall_avg↑ (per-trial precision)", color="#e0e0e0", fontsize=11)
    ax1.set_title("3B: Diversity Loss — recall_avg vs Diversity", color="#e0e0e0", fontsize=12)
    ax1.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=8)

    fig.suptitle("Sub-exp 3B: Diversity Adapter Training — MovieLens 1M (λ=1.0)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "diversity_adapter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved → {path}")


# ── 統合トレードオフ図 (plan_002 と横並びで比較) ───────────────────

def plot_combined_003(all_results: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#e0e0e0")
    ax.spines[:].set_color("#2d3142")
    ax.grid(True, color="#2d3142", linewidth=0.5)

    for name, r in all_results.items():
        div = 1 - r["mean"]["temporal_overlap"]
        rc  = r["mean"]["recall_cum"]

        if name.startswith("3A_"):
            parts = name.split("_")  # ['3A', 'input', 'sXpXXX']
            pos   = parts[1]
            color = {"input": "#e74c3c", "middle": "#f39c12", "output": "#2ecc71"}.get(pos, "#888")
            marker = {"input": "o", "middle": "s", "output": "^"}.get(pos, "o")
            ax.scatter(div, rc, color=color, marker=marker, s=80, alpha=0.8, zorder=3)
            ax.annotate(name.replace("3A_", ""), (div, rc),
                        textcoords="offset points", xytext=(3, 2), color="#aaa", fontsize=6)
        elif name.startswith("3B_"):
            loss = name.split("_")[1] if "_" in name else name
            color = DIV_COLORS.get(name.replace("3B_", "").replace("_l1p0", ""), "#888")
            ax.scatter(div, rc, color=color, marker="D", s=100, alpha=0.9, zorder=4)
            ax.annotate(name.replace("3B_", ""), (div, rc),
                        textcoords="offset points", xytext=(3, 2), color="#ddd", fontsize=6)

    # 凡例
    for pos, color, marker, label in [
        ("input",  "#e74c3c", "o", "3A Input noise"),
        ("middle", "#f39c12", "s", "3A Middle noise"),
        ("output", "#2ecc71", "^", "3A Output noise"),
    ]:
        ax.scatter([], [], color=color, marker=marker, s=80, label=label)
    for loss_name, color in DIV_COLORS.items():
        ax.scatter([], [], color=color, marker="D", s=80, label=f"3B {loss_name}")

    ax.set_xlabel("Temporal Diversity (1 - Overlap) ↑", color="#e0e0e0", fontsize=12)
    ax.set_ylabel("Recall@10 Cumulative ↑", color="#e0e0e0", fontsize=12)
    ax.set_title("Plan 003: All Models — Precision-Diversity Tradeoff",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=8,
              ncol=2, loc="lower right")
    plt.tight_layout()
    path = out_dir / "tradeoff_003.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main(subexp: str, device: str, n_trials: int = 5):
    processed_dir = Path("data/processed/movielens")
    report_dir    = Path("report/plan_003")
    report_dir.mkdir(parents=True, exist_ok=True)

    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(processed_dir)
    index     = build_index(item_embs)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt   = get_test_gt(interactions, uid2idx, iid2idx)

    # user_texts の読み込み (属性テキスト) — user_idx 順に揃える
    import pandas as _pd
    uid_map  = _pd.read_parquet(processed_dir / "user_id_map.parquet")   # user_id → index
    user_df  = _pd.read_parquet(processed_dir / "user_texts.parquet")
    merged   = uid_map.merge(user_df[["user_id", "user_text"]], on="user_id")
    merged   = merged.sort_values("index")
    user_texts = [f"query: {t}" for t in merged["user_text"].tolist()]
    log.info(f"Loaded {len(user_texts)} user texts (sorted by user_idx)")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    all_results: dict = {}

    # ─── Sub-exp 3A ──────────────────────────────────────────────────
    if subexp in ("3a", "all"):
        log.info("\n" + "="*70 + "\nSub-exp 3A: Noise position sweep\n" + "="*70)
        out_dir = report_dir / "noise_position"
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Precomputing all noisy embeddings (output=numpy fast, input/middle=mE5) ...")
        trial_cache = precompute_all_noisy_embeddings(
            user_texts, dev, n_trials=n_trials, batch_size=256,
            clean_user_embs=user_embs,  # output位置はnumpy加算で高速化
        )
        log.info("Precompute done.")

        models = [
            M_NoisePosition(pos, sigma, trial_cache[(pos, sigma)])
            for pos in POSITIONS_3A
            for sigma in SIGMAS_3A
        ]

        res = run_models(models, test_gt, index, item_embs, train_pos, user_embs, device, len(item_embs))
        save_results(res, out_dir)
        plot_noise_position(res, out_dir)
        all_results.update(res)

    # ─── Sub-exp 3B ──────────────────────────────────────────────────
    if subexp in ("3b", "all"):
        log.info("\n" + "="*70 + "\nSub-exp 3B: Diversity adapter training\n" + "="*70)
        out_dir = report_dir / "diversity_adapter"
        out_dir.mkdir(parents=True, exist_ok=True)

        models = make_diversity_adapter_models(lambda_div=1.0)
        res = run_models(models, test_gt, index, item_embs, train_pos, user_embs, device, len(item_embs))
        save_results(res, out_dir)
        plot_diversity_adapter(res, out_dir)
        all_results.update(res)

    # ─── 統合プロット ─────────────────────────────────────────────────
    if all_results:
        with open(report_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        plot_combined_003(all_results, report_dir)
        log.info(f"\nAll results saved → {report_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all", choices=["all", "3a", "3b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_trials", type=int, default=5)
    args = parser.parse_args()
    main(args.subexp, args.device, n_trials=args.n_trials)
