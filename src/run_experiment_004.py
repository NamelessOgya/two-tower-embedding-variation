"""
Plan 004 Experiment Runner
--------------------------
目的:
  全手法の精度-多様性トレードオフ曲線を一枚のグラフで比較する。

4A: 3B Diversity Adapter の λ スイープ
    loss_fns  = [cosine_emb, l2_emb, soft_jaccard]
    lambdas   = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    λ=0.0 は純粋 BPR（多様性なし）のコントロール

4B: 全手法統合プロット
    - M0 baseline（多様性なし、単点）
    - M4 Gaussian σ スイープ（plan_002 の既存データを読み込み）
    - 3B λ スイープ（今回の新規実験 4A の結果）

Usage:
  python src/run_experiment_004.py --subexp all
  python src/run_experiment_004.py --subexp 4a
  python src/run_experiment_004.py --subexp plot_only  # 既存データのみでプロット
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.run_experiment import (
    load_data, build_index, get_train_pos, get_test_gt,
    evaluate_one_seed, SEEDS, K, N_TRIALS,
)
from src.model.models_003 import M_DiversityAdapter, DIV_LOSSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 設定 ─────────────────────────────────────────────────────────────
LAMBDA_SWEEP   = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
SWEEP_LOSSES   = ["cosine_emb", "l2_emb", "kl_dist", "js_dist", "soft_jaccard", "listnet"]  # 全6種

# 比較に使う M4 σ スイープのラベル（plan_002 の結果 JSON のキー）
M4_RESULT_PATH  = Path("report/plan_002/sigma_sweep/results.json")
M0_RECALL_CUM   = 0.0016   # M0 baseline の recall_cum
M0_RECALL_AVG   = 0.0016   # M0 baseline の recall_avg
M0_OVERLAP      = 1.0000   # M0 baseline の temporal_overlap


# ── 評価ループ ─────────────────────────────────────────────────────────

def run_models(models, test_gt, index, item_embs, train_pos, user_embs, device, n_total_items):
    results = {}
    for model in models:
        log.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        t0 = time.time()
        model.prepare(train_pos, user_embs, item_embs, device=device)
        log.info(f"  prepare: {time.time()-t0:.1f}s")

        seed_results = []
        for seed in SEEDS:
            m = evaluate_one_seed(
                model, test_gt, index, item_embs,
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


# ── 統合トレードオフ曲線の描画 ────────────────────────────────────────

# パレット
LOSS_COLORS = {
    "cosine_emb":   "#f4d03f",
    "l2_emb":       "#e67e22",
    "kl_dist":      "#3498db",
    "js_dist":      "#9b59b6",
    "soft_jaccard": "#e74c3c",
    "listnet":      "#1abc9c",
}
LOSS_LABELS = {
    "cosine_emb":   "3B cosine_emb (λ sweep)",
    "l2_emb":       "3B l2_emb (λ sweep)",
    "kl_dist":      "3B kl_dist (λ sweep)",
    "js_dist":      "3B js_dist (λ sweep)",
    "soft_jaccard": "3B soft_jaccard (λ sweep)",
    "listnet":      "3B listnet (λ sweep)",
}


def _ax_style(ax, title, xlabel, ylabel):
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#2d3142")
    ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.7)
    ax.set_title(title, color="#e0e0e0", fontsize=12)
    ax.set_xlabel(xlabel, color="#e0e0e0", fontsize=10)
    ax.set_ylabel(ylabel, color="#e0e0e0", fontsize=10)


def plot_unified_tradeoff(
    new_results: dict,
    out_dir: Path,
    y_metric: str = "recall_avg",   # "recall_avg" or "recall_cum"
):
    """
    全手法の精度-多様性トレードオフ曲線を2枚（recall_avg, recall_cum）描画。
    diversity = 1 - temporal_overlap を X 軸とする。
    """

    # ── M4 Gaussian の既存データを読み込む ────────────────────────────
    m4_points = []   # list of (diversity, recall_avg, recall_cum, sigma)
    if M4_RESULT_PATH.exists():
        m4_data = json.load(open(M4_RESULT_PATH))
        for key, r in m4_data.items():
            if not key.startswith("M4_gauss_sigma"):
                continue
            m = r["mean"]
            sigma_str = key.replace("M4_gauss_sigma", "").replace("p", ".")
            try:
                sigma = float(sigma_str)
            except ValueError:
                continue
            div = 1.0 - m["temporal_overlap"]
            ra  = m.get("recall_avg", m.get("recall_single", 0.0))
            rc  = m["recall_cum"]
            m4_points.append((div, ra, rc, sigma))
        m4_points.sort(key=lambda x: x[0])
    else:
        log.warning(f"M4 result not found: {M4_RESULT_PATH}")

    # ── 3B λ スイープの新規データを整理 ────────────────────────────────
    # {loss_name: [(div, recall_avg, recall_cum, lambda), ...]}
    adapter_curves: dict[str, list] = {loss: [] for loss in SWEEP_LOSSES}
    for name, r in new_results.items():
        for loss in SWEEP_LOSSES:
            prefix = f"3B_{loss}_l"
            if name.startswith(prefix):
                lam_str = name[len(prefix):].replace("p", ".")
                try:
                    lam = float(lam_str)
                except ValueError:
                    continue
                m = r["mean"]
                div = 1.0 - m["temporal_overlap"]
                ra  = m.get("recall_avg", m.get("recall_single", 0.0))
                rc  = m["recall_cum"]
                adapter_curves[loss].append((div, ra, rc, lam))
    for loss in SWEEP_LOSSES:
        adapter_curves[loss].sort(key=lambda x: x[0])

    # ── 描画: recall_avg vs diversity AND recall_cum vs diversity ──────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0f1117")
    titles = [
        "Precision-Diversity Tradeoff (recall_avg = per-trial precision)",
        "Cumulative-Diversity Tradeoff (recall_cum = N-trial cumulative)",
    ]
    ylabels = ["recall_avg↑ (1試行あたり精度)", "recall_cum↑ (N試行累積 recall)"]
    y_indices = [1, 2]   # index in (div, ra, rc, param)

    for ax, title, ylabel, yi in zip(axes, titles, ylabels, y_indices):
        _ax_style(ax, title, "Diversity (1 − temporal_overlap) ↑", ylabel)

        # M0 baseline（単点）
        ax.scatter([0.0], [M0_RECALL_AVG if yi == 1 else M0_RECALL_CUM],
                   color="#ffffff", marker="*", s=250, zorder=10, label="M0 baseline (no diversity)")
        ax.annotate("M0", (0.0, M0_RECALL_AVG if yi == 1 else M0_RECALL_CUM),
                    color="#ccc", fontsize=8, xytext=(4, 4), textcoords="offset points")

        # M4 Gaussian σ スイープ（曲線）
        if m4_points:
            divs = [p[0] for p in m4_points]
            ys   = [p[yi] for p in m4_points]
            sigs = [p[3] for p in m4_points]
            ax.plot(divs, ys, "-", color="#3498db", lw=2.0, alpha=0.9, zorder=4,
                    label="M4 Gaussian (σ sweep)")
            ax.scatter(divs, ys, color="#3498db", s=50, zorder=5, edgecolors="white", linewidths=0.3)
            for d, y, s in zip(divs, ys, sigs):
                ax.annotate(f"σ={s:.3f}", (d, y), fontsize=6, color="#aaddff",
                            xytext=(3, 3), textcoords="offset points")

        # 3B アダプタ λ スイープ（各 loss_fn ごとに曲線）
        for loss in SWEEP_LOSSES:
            pts = adapter_curves[loss]
            if not pts:
                continue
            divs = [p[0] for p in pts]
            ys   = [p[yi] for p in pts]
            lams = [p[3] for p in pts]
            color = LOSS_COLORS[loss]
            label = LOSS_LABELS[loss]
            ax.plot(divs, ys, "--", color=color, lw=2.0, alpha=0.9, zorder=5, label=label)
            ax.scatter(divs, ys, color=color, s=70, zorder=6,
                       edgecolors="white", linewidths=0.4, marker="D")
            for d, y, lam in zip(divs, ys, lams):
                ax.annotate(f"λ={lam}", (d, y), fontsize=6, color=color,
                            xytext=(3, -8), textcoords="offset points")

        ax.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0",
                  fontsize=8, loc="upper left")

    fig.suptitle(
        "Plan 004: Unified Precision-Diversity Tradeoff — All Methods — MovieLens 1M",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "tradeoff_unified.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main(subexp: str, device: str, n_trials: int = 5):
    processed_dir = Path("data/processed/movielens")
    report_dir    = Path("report/plan_004")
    report_dir.mkdir(parents=True, exist_ok=True)

    new_results: dict = {}

    # ─── Sub-exp 4A: λ スイープ ────────────────────────────────────────
    if subexp in ("4a", "all"):
        log.info("\n" + "="*70 + "\nSub-exp 4A: λ sweep for 3B Diversity Adapters\n" + "="*70)

        interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(processed_dir)
        index     = build_index(item_embs)
        train_pos = get_train_pos(interactions, uid2idx, iid2idx)
        test_gt   = get_test_gt(interactions, uid2idx, iid2idx)

        import torch
        dev = torch.device(device if torch.cuda.is_available() else "cpu")

        models = [
            M_DiversityAdapter(loss_name, lambda_div=lam)
            for loss_name in SWEEP_LOSSES
            for lam in LAMBDA_SWEEP
        ]
        log.info(f"  Models: {len(models)} ({len(SWEEP_LOSSES)} losses × {len(LAMBDA_SWEEP)} λ values)")

        out_dir = report_dir / "lambda_sweep"
        res = run_models(models, test_gt, index, item_embs, train_pos, user_embs, dev, len(item_embs))
        save_results(res, out_dir)
        new_results.update(res)

        # λ スイープ曲線のみを単独プロット
        _plot_lambda_sweep(res, report_dir)

    # ─── Sub-exp 4B: 統合プロット（既存 + 新規） ──────────────────────
    if subexp in ("4b", "all", "plot_only"):
        # 既存の 3B λ=1.0 結果も統合
        p003_3b = Path("report/plan_003/diversity_adapter/results.json")
        if p003_3b.exists():
            d = json.load(open(p003_3b))
            for k, v in d.items():
                if k not in new_results:
                    new_results[k] = v

        # 新規の λ スイープ結果を追加（4A を skip した場合に既存ファイルから読む）
        lsweep_path = report_dir / "lambda_sweep" / "results.json"
        if lsweep_path.exists() and subexp in ("4b", "plot_only"):
            d = json.load(open(lsweep_path))
            new_results.update(d)

        # 統合 JSON 保存
        with open(report_dir / "all_results.json", "w") as f:
            json.dump(new_results, f, indent=2)

        plot_unified_tradeoff(new_results, report_dir)
        log.info(f"\nAll plots saved → {report_dir}/tradeoff_unified.png")


def _plot_lambda_sweep(results: dict, out_dir: Path):
    """λ スイープの各 loss_fn を 1 行で見せる個別プロット。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.7)

    ax0, ax1 = axes

    for loss in SWEEP_LOSSES:
        pts = []
        for lam in LAMBDA_SWEEP:
            name = f"3B_{loss}_l{lam:.1f}".replace(".", "p")
            if name not in results:
                continue
            m = results[name]["mean"]
            div = 1.0 - m["temporal_overlap"]
            ra  = m.get("recall_avg", m.get("recall_single", 0.0))
            rc  = m["recall_cum"]
            pts.append((lam, div, ra, rc))

        if not pts:
            continue

        color = LOSS_COLORS[loss]
        label = loss

        # ax0: λ vs recall_avg
        lams = [p[0] for p in pts]
        ras  = [p[2] for p in pts]
        ax0.plot(lams, ras, "o-", color=color, lw=2, label=label)

        # ax1: precision-diversity (recall_avg vs diversity)
        divs = [p[1] for p in pts]
        ax1.plot(divs, ras, "D--", color=color, lw=2, label=label)
        for d, ra, lam in zip(divs, ras, lams):
            ax1.annotate(f"λ={lam}", (d, ra), fontsize=6, color=color,
                         xytext=(3, 2), textcoords="offset points")

    ax0.set_xlabel("λ (diversity loss weight)", color="#e0e0e0", fontsize=10)
    ax0.set_ylabel("recall_avg↑", color="#e0e0e0", fontsize=10)
    ax0.set_title("4A: λ → recall_avg", color="#e0e0e0", fontsize=12)
    ax0.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=8)

    ax1.set_xlabel("Diversity (1−overlap) ↑", color="#e0e0e0", fontsize=10)
    ax1.set_ylabel("recall_avg↑", color="#e0e0e0", fontsize=10)
    ax1.set_title("4A: λ sweep — Precision-Diversity Frontier", color="#e0e0e0", fontsize=12)
    ax1.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#e0e0e0", fontsize=8)

    # M0 baseline
    ax0.axhline(M0_RECALL_AVG, color="#aaa", ls=":", lw=1, label="M0 baseline")
    ax1.scatter([0.0], [M0_RECALL_AVG], color="white", marker="*", s=150, zorder=10,
                label="M0 baseline")

    fig.suptitle("Sub-exp 4A: λ Sweep — Diversity Adapter", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "lambda_sweep.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all", choices=["all", "4a", "4b", "plot_only"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_trials", type=int, default=5)
    args = parser.parse_args()
    main(args.subexp, args.device, n_trials=args.n_trials)
