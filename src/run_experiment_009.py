"""
Plan 009 Experiment Runner
---------------------------
Sub-exp 9A: soft_jaccard λ 精密スイープ (λ ∈ {0.001, 0.005, 0.01, 0.03, 0.05, 0.1})
Sub-exp 9B: kl_dist バグ修正版再実験 (λ ∈ {0.1, 0.5, 1.0}, plan_008 との比較)
Sub-exp 9C: 統合 Pareto プロット（Plan 008 ベースラインと合わせて描画）

Usage:
    PYTHONPATH=. python3 src/run_experiment_009.py --subexp all --device cuda
    PYTHONPATH=. python3 src/run_experiment_009.py --subexp 9A --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.run_experiment import (
    load_data, get_train_pos, get_test_gt,
    SEEDS, K, N_TRIALS,
)
from src.evaluate.metrics import (
    recall_at_k, recall_at_k_single, hit_at_k, ndcg_at_k,
    temporal_overlap_rate, intra_list_diversity, coverage,
)
from src.model.models_007 import TwoTowerModel
from src.model.models_008 import TwoTowerDivLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Sweep Parameters ───────────────────────────────────────────────────────────
BEST_DEPTH      = 2
BEST_HIDDEN_DIM = 64

# Sub-exp 9A: soft_jaccard 精密スイープ
LAMBDA_9A = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
SIGMA_DIV = 0.05

# Sub-exp 9B: kl_dist 修正版
LAMBDA_9B = [0.1, 0.5, 1.0]

N_SEEDS = len(SEEDS)

OUT_DIR  = Path("report/plan_009")
DATA_DIR = Path("data/processed/movielens")
PLAN_008_RESULTS = Path("report/plan_008/results.json")


# ── Evaluation Helpers ─────────────────────────────────────────────────────────

def evaluate_model_seed(
    model,
    test_gt: dict,
    index: faiss.IndexFlatIP,
    item_embs: np.ndarray,
    k: int,
    n_trials: int,
    seed: int,
    n_total_items: int,
) -> dict:
    rng = np.random.default_rng(seed)
    per_user: dict = defaultdict(list)
    global_recommended: set = set()

    for user_idx, gt in test_gt.items():
        trial_lists = []
        trial_sets  = []
        for trial in range(n_trials):
            recs = model.recommend(user_idx, trial, rng, index, k)
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

    mean_r = {k2: float(np.mean(v)) for k2, v in per_user.items()}
    mean_r["coverage"] = len(global_recommended) / n_total_items
    return mean_r


def run_model_eval(model, test_gt, train_pos, user_embs, item_embs, device, n_total_items) -> dict:
    log.info("\n" + "="*60 + f"\nModel: {model.name}\n" + "="*60)
    model.prepare(train_pos, user_embs, item_embs, device=device)
    index = model.build_index()

    seed_results = []
    for seed in range(N_SEEDS):
        r = evaluate_model_seed(
            model, test_gt, index, item_embs, K, N_TRIALS, seed, n_total_items
        )
        log.info(f"  seed={seed}  rc={r['recall_cum']:.4f}  ra={r['recall_avg']:.4f}  "
                 f"hit={r['hit']:.4f}  ov={r['temporal_overlap']:.4f}  cov={r['coverage']:.4f}")
        seed_results.append(r)

    keys = seed_results[0].keys()
    mean = {k2: float(np.mean([r[k2] for r in seed_results])) for k2 in keys}
    std  = {k2: float(np.std( [r[k2] for r in seed_results])) for k2 in keys}
    log.info(f"  [AVG] rc={mean['recall_cum']:.4f}±{std['recall_cum']:.4f}  "
             f"ra={mean['recall_avg']:.4f}±{std['recall_avg']:.4f}  "
             f"hit={mean['hit']:.4f}±{std['hit']:.4f}  "
             f"ov={mean['temporal_overlap']:.4f}±{std['temporal_overlap']:.4f}")
    return {"mean": mean, "std": std}


def save_results(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    rows = []
    for name, r in all_results.items():
        rows.append({
            "model": name,
            **{f"{k}_mean": v for k, v in r["mean"].items()},
            **{f"{k}_std":  v for k, v in r["std"].items()},
        })
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)


# ── Sub-exp 9A: soft_jaccard λ 精密スイープ ──────────────────────────────────

def run_9a(base_tt: TwoTowerModel, train_pos, test_gt, user_embs, item_embs, device, n_total_items) -> dict:
    log.info("\n" + "="*60 + "\nSub-exp 9A: soft_jaccard lambda fine sweep\n" + "="*60)
    results = {}
    for lam in LAMBDA_9A:
        model = TwoTowerDivLoss(
            base_tt=base_tt,
            div_loss_name="soft_jaccard",
            lambda_div=lam,
            sigma=SIGMA_DIV,
            lr=2e-3,
            epochs=30,
            batch_size=512,
        )
        # 既存結果があればスキップ
        results[model.name] = run_model_eval(
            model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
    return results


# ── Sub-exp 9B: kl_dist バグ修正版 ────────────────────────────────────────────

def run_9b(base_tt: TwoTowerModel, train_pos, test_gt, user_embs, item_embs, device, n_total_items) -> dict:
    log.info("\n" + "="*60 + "\nSub-exp 9B: kl_dist (bugfixed) sweep\n" + "="*60)
    results = {}
    for lam in LAMBDA_9B:
        model = TwoTowerDivLoss(
            base_tt=base_tt,
            div_loss_name="kl_dist",
            lambda_div=lam,
            sigma=SIGMA_DIV,
            lr=2e-3,
            epochs=30,
            batch_size=512,
        )
        # バグ修正版であることを名前で区別
        model.name = model.name.replace("TT_divloss_kl_dist", "TT_divloss_kl_dist_fixed")
        results[model.name] = run_model_eval(
            model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
    return results


# ── Sub-exp 9C: 統合プロット ──────────────────────────────────────────────────

def plot_unified_009(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    FAMILIES = [
        # Plan 008 ベースライン（参照） ── より具体的なプレフィックスを先に並べる
        ("M0_strong",                    "#aaaaaa", "o", 10, "M0_strong",                None),
        # postnoise を base TT より先に置く（claimed の先取りを防ぐ）
        ("TwoTower_d2_h64_postnoise",    "#e74c3c", "v",  9, "PostNoise (sigma sweep)",   "sigma"),
        ("TwoTower_d2_h64",              "#3498db", "s", 10, "TT_d2_h64 (no div)",       None),
        ("TT_soft_jaccard_l0p5",         "#9b59b6", "^",  9, "TT+soft_jaccard (008)",    None),
        # Plan 008 soft_jaccard (λ=0.1 以上)
        ("TT_divloss_soft_jaccard_l0p1", "#f39c12", "*", 10, "DivLoss soft_jaccard (008 λ=0.1)", None),
        # Plan 009 soft_jaccard 精密スイープ（9A）── l0p1 より先に具体的プレフィックスを除外済み
        ("TT_divloss_soft_jaccard",      "#f1c40f", "*", 10, "DivLoss soft_jaccard (009 fine)", "lambda"),
        # Plan 008 kl_dist（バグ版）
        ("TT_divloss_kl_dist_l",         "#95a5a6", "x",  8, "DivLoss kl_dist (008 buggy)", "lambda"),
        # Plan 009 kl_dist 修正版（9B）
        ("TT_divloss_kl_dist_fixed",     "#e74c3c", "X", 10, "DivLoss kl_dist (009 fixed)", "lambda"),
    ]

    PARAM_MAP = {
        "l0p001": "λ=0.001", "l0p005": "λ=0.005",
        "l0p01": "λ=0.01",  "l0p03": "λ=0.03",
        "l0p05": "λ=0.05",  "l0p1": "λ=0.1",
        "l0p5": "λ=0.5",    "l1p0": "λ=1.0",
        "s0p05": "σ=0.05",  "s0p1": "σ=0.10",
    }

    def get_param_label(name: str) -> str:
        for k, v in PARAM_MAP.items():
            if name.endswith(k):
                return v
        return name.split("_")[-1]

    data = {}
    for name, r in all_results.items():
        m = r["mean"]
        data[name] = {
            "div": 1.0 - m["temporal_overlap"],
            "rc":  m["recall_cum"],
            "ra":  m["recall_avg"],
        }

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8), facecolor="#0f1117")
    for ax in (ax0, ax1):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)

    # 各モデルを「最初にマッチしたファミリー」にのみ割り当て（ダブルプロット防止）
    claimed: set[str] = set()
    family_members: dict[str, list] = {}
    for prefix, color, marker, size, label, _ in FAMILIES:
        members = sorted(
            [(n, d) for n, d in data.items()
             if n not in claimed and (n == prefix or n.startswith(prefix + "_"))],
            key=lambda x: x[1]["div"]
        )
        family_members[prefix] = members
        for n, _ in members:
            claimed.add(n)

    for prefix, color, marker, size, label, _ in FAMILIES:
        members = family_members.get(prefix, [])
        if not members:
            continue

        divs = [m[1]["div"] for m in members]
        rcs  = [m[1]["rc"]  for m in members]
        ras  = [m[1]["ra"]  for m in members]

        line_kw = dict(color=color, linewidth=1.3, alpha=0.55, zorder=1)
        if len(members) >= 2:
            ax0.plot(divs, rcs, **line_kw)
            ax1.plot(divs, ras, **line_kw)

        for i, (name, d) in enumerate(members):
            lbl = label if i == 0 else "_nolegend_"
            ax0.scatter(d["div"], d["rc"], c=color, marker=marker, s=size**2,
                        alpha=0.92, label=lbl, edgecolors="white", linewidths=0.4, zorder=2)
            ax1.scatter(d["div"], d["ra"], c=color, marker=marker, s=size**2,
                        alpha=0.92, label=lbl, edgecolors="white", linewidths=0.4, zorder=2)
            ann = get_param_label(name)
            ax0.annotate(ann, (d["div"], d["rc"]), fontsize=6.5, color="#dddddd",
                         xytext=(4, 4), textcoords="offset points")
            ax1.annotate(ann, (d["div"], d["ra"]), fontsize=6.5, color="#dddddd",
                         xytext=(4, 4), textcoords="offset points")


    xlabel = "Diversity (1 - temporal_overlap)"
    for ax, ylabel, title in [
        (ax0, "recall_cum  (N-trial cumulative Recall)",  "Diversity vs recall_cum  [Plan 009]"),
        (ax1, "recall_avg  (per-trial mean Recall)",      "Diversity vs recall_avg  [Plan 009]"),
    ]:
        ax.set_xlabel(xlabel, color="#e0e0e0", fontsize=12)
        ax.set_ylabel(ylabel, color="#e0e0e0", fontsize=12)
        ax.set_title(title, color="#ffffff", fontsize=14, pad=12, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#555",
                  labelcolor="#e0e0e0", loc="best", framealpha=0.8)

    plt.tight_layout(pad=2.0)
    out_path = out_dir / "tradeoff_009_unified.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info(f"Saved plot -> {out_path}")


def print_summary(all_results: dict):
    rows = []
    for name, r in all_results.items():
        m, s = r["mean"], r["std"]
        rows.append({
            "model":            name,
            "recall_cum":       f"{m['recall_cum']:.4f}±{s['recall_cum']:.4f}",
            "recall_avg":       f"{m['recall_avg']:.4f}±{s['recall_avg']:.4f}",
            "hit":              f"{m['hit']:.4f}±{s['hit']:.4f}",
            "temporal_overlap": f"{m['temporal_overlap']:.4f}±{s['temporal_overlap']:.4f}",
        })
    log.info("\n" + pd.DataFrame(rows).to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all",
                        choices=["all", "9A", "9B", "9C"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_all = args.subexp == "all"

    # データロード
    log.info("Loading MovieLens 1M ...")
    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(DATA_DIR)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt   = get_test_gt(interactions, uid2idx, iid2idx)
    n_total_items = len(item_embs)
    log.info(f"  users={len(train_pos)}, items={n_total_items}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 既存結果ロード（再開対応）
    results_path = OUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        log.info(f"Loaded {len(all_results)} existing results")
    else:
        all_results = {}

    # Plan 008 ベースラインを参照としてロード
    if PLAN_008_RESULTS.exists():
        with open(PLAN_008_RESULTS) as f:
            r008 = json.load(f)
        baseline_keys = [
            "M0_strong", "TwoTower_d2_h64",
            "TwoTower_d2_h64_postnoise_s0p05", "TwoTower_d2_h64_postnoise_s0p1",
            "TT_soft_jaccard_l0p5",
            "TT_divloss_soft_jaccard_l0p1_s0p05",
            "TT_divloss_kl_dist_l0p1_s0p05",
            "TT_divloss_kl_dist_l0p5_s0p05",
            "TT_divloss_kl_dist_l1p0_s0p05",
        ]
        for k in baseline_keys:
            if k in r008 and k not in all_results:
                all_results[k] = r008[k]
        save_results(all_results, OUT_DIR)
        log.info(f"Loaded baselines from Plan 008")

    # best Two-Tower を再学習
    def rebuild_best_tt() -> TwoTowerModel:
        tt = TwoTowerModel(
            hidden_dim=BEST_HIDDEN_DIM, depth=BEST_DEPTH,
            lr=1e-3, epochs=50, batch_size=1024,
            logit_scale=14.3, alpha=0.1,
        )
        log.info(f"Re-training TwoTower_d{BEST_DEPTH}_h{BEST_HIDDEN_DIM} ...")
        tt.prepare(train_pos, user_embs, item_embs, device=args.device)
        return tt

    # 9A / 9B は同じ base_tt を使うので 1 回だけ学習
    if run_all or args.subexp in ("9A", "9B"):
        best_tt = rebuild_best_tt()

        if run_all or args.subexp == "9A":
            r = run_9a(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

        if run_all or args.subexp == "9B":
            r = run_9b(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

    if run_all or args.subexp == "9C":
        print_summary(all_results)
        plot_unified_009(all_results, OUT_DIR)

    log.info("\n✅ Plan 009 completed!")


if __name__ == "__main__":
    main()
