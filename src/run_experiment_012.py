"""
Plan 012 Experiment Runner
---------------------------
Advanced Soft-Jaccard Variants (12A: TopK, 12B: Adaptive, 12C: Semantic)
および 12E: Item Partition ベースラインを評価し、
Plan 009〜011 の最良モデル群と比較統合する。

Usage:
    PYTHONPATH=. python3 src/run_experiment_012.py --subexp all --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.run_experiment import load_data, get_train_pos, get_test_gt, SEEDS, K, N_TRIALS
from src.evaluate.metrics import (
    recall_at_k, recall_at_k_single, hit_at_k, ndcg_at_k,
    temporal_overlap_rate,
)
from src.model.models_007 import TwoTowerModel
from src.model.models_008 import TwoTowerDivLoss
from src.model.models_012 import (
    TwoTowerTopKSoftJaccard,
    TwoTowerAdaptiveSoftJaccard,
    TwoTowerSemanticSoftJaccard,
    TwoTowerItemPartition,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BEST_DEPTH      = 2
BEST_HIDDEN_DIM = 64
N_SEEDS         = len(SEEDS)

OUT_DIR          = Path("report/plan_012")
DATA_DIR         = Path("data/processed/movielens")
PLAN_009_RESULTS = Path("report/plan_009/results.json")
PLAN_010_RESULTS = Path("report/plan_010/results.json")
PLAN_011_RESULTS = Path("report/plan_011/results.json")

TOPK_CONFIGS = [
    (30, 0.1, 0.05),
    (30, 0.3, 0.05),
    (10, 0.1, 0.05),
]

ADAPTIVE_CONFIGS = [
    (0.1, 0.05),
    (0.3, 0.05),
]

SEMANTIC_CONFIGS = [
    (0.1, 0.05),
    (0.3, 0.05),
]


def run_model_eval(model, test_gt, train_pos, user_embs, item_embs, device, n_total_items):
    model.prepare(train_pos, user_embs, item_embs, device=device)
    index = model.build_index()

    seed_results = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        per_user = defaultdict(list)
        for user_idx, gt in test_gt.items():
            trial_lists = []
            trial_sets  = []
            for trial in range(N_TRIALS):
                recs = model.recommend(user_idx, trial, rng, index, K)
                trial_lists.append(recs)
                trial_sets.append(set(recs))
            per_user["recall_cum"].append(recall_at_k(trial_sets, gt))
            per_user["recall_avg"].append(
                float(np.mean([recall_at_k_single(s, gt) for s in trial_sets]))
            )
            per_user["hit"].append(
                float(np.mean([hit_at_k(s, gt) for s in trial_sets]))
            )
            per_user["ndcg"].append(
                float(np.mean([ndcg_at_k(lst, gt, K) for lst in trial_lists]))
            )
            per_user["temporal_overlap"].append(temporal_overlap_rate(trial_sets, K))

        mean_r = {k2: float(np.mean(v)) for k2, v in per_user.items()}
        std_r  = {k2: float(np.std(v))  for k2, v in per_user.items()}
        log.info(
            f"  seed={seed}  rc={mean_r['recall_cum']:.4f}  "
            f"ra={mean_r['recall_avg']:.4f}  "
            f"ov={mean_r['temporal_overlap']:.4f}"
        )
        seed_results.append(mean_r)

    keys = seed_results[0].keys()
    mean = {k2: float(np.mean([r[k2] for r in seed_results])) for k2 in keys}
    std  = {k2: float(np.std( [r[k2] for r in seed_results])) for k2 in keys}
    raw  = {k2: [r[k2] for r in seed_results] for k2 in keys}
    log.info(
        f"  [AVG] rc={mean['recall_cum']:.4f}±{std['recall_cum']:.4f}  "
        f"ra={mean['recall_avg']:.4f}±{std['recall_avg']:.4f}  "
        f"ov={mean['temporal_overlap']:.4f}±{std['temporal_overlap']:.4f}"
    )
    return {"mean": mean, "std": std, "raw": raw}


def save_results(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)


def load_baselines() -> dict:
    baselines = {}
    for path in [PLAN_009_RESULTS, PLAN_010_RESULTS, PLAN_011_RESULTS]:
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        for k in [
            "TwoTower_d2_h64",
            "TwoTower_d2_h64_postnoise_s0p1",
            "TwoTower_d2_h64_postnoise_s0p2",
            "TT_divloss_soft_jaccard_l0p1_s0p05",
            "TwoTower_d2_h64_dpp_s0p1",
            "TT_multihead_M3_l0p1_s0p05",
        ]:
            if k in d and k not in baselines:
                baselines[k] = d[k]
    return baselines


def run_12a(base_tt, train_pos, test_gt, user_embs, item_embs, device, n_total_items):
    log.info("\n" + "="*60 + "\nSub-exp 12A: Top-K Truncated Soft Jaccard\n" + "="*60)
    results = {}
    for topk, lam, sig in TOPK_CONFIGS:
        model = TwoTowerTopKSoftJaccard(
            base_tt=base_tt, topk=topk, lambda_div=lam, sigma=sig
        )
        log.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        results[model.name] = run_model_eval(
            model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
    return results


def run_12b(base_tt, train_pos, test_gt, user_embs, item_embs, device, n_total_items):
    log.info("\n" + "="*60 + "\nSub-exp 12B: Learned Per-Dimension Noise (Adaptive Soft Jaccard)\n" + "="*60)
    results = {}
    for lam, init_sig in ADAPTIVE_CONFIGS:
        model = TwoTowerAdaptiveSoftJaccard(
            base_tt=base_tt, lambda_div=lam, init_sigma=init_sig
        )
        log.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        results[model.name] = run_model_eval(
            model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
    return results


def run_12c(base_tt, train_pos, test_gt, user_embs, item_embs, device, n_total_items):
    log.info("\n" + "="*60 + "\nSub-exp 12C: Semantic-aware Soft Jaccard\n" + "="*60)
    results = {}
    for lam, sig in SEMANTIC_CONFIGS:
        model = TwoTowerSemanticSoftJaccard(
            base_tt=base_tt, lambda_div=lam, sigma=sig
        )
        log.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        results[model.name] = run_model_eval(
            model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
    return results


def run_12e(base_tt, train_pos, test_gt, user_embs, item_embs, device, n_total_items):
    log.info("\n" + "="*60 + "\nSub-exp 12E: Item Partition Baseline (全アイテム重複ゼロ分割)\n" + "="*60)
    results = {}
    model = TwoTowerItemPartition(base_tt=base_tt, n_trials=N_TRIALS)
    log.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
    results[model.name] = run_model_eval(
        model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
    )
    return results


def plot_12d(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    FAMILIES = [
        # ── ベースライン（Plan 009〜011）
        ("TwoTower_d2_h64",                       "#555555", "o",  9, "TT no-div",             None),
        ("TwoTower_d2_h64_postnoise_s0p1",        "#e74c3c", "v",  8, "PostNoise σ=0.10",       None),
        ("TwoTower_d2_h64_postnoise_s0p2",        "#e74c3c", "v",  8, "PostNoise σ=0.20",       None),
        ("TT_divloss_soft_jaccard_l0p1_s0p05",    "#f1c40f", "*", 12, "SoftJaccard λ=0.1",      None),
        ("TwoTower_d2_h64_dpp_s0p1",              "#3498db", "D",  9, "DPP (base TT)",          None),
        ("TT_multihead_M3_l0p1_s0p05",            "#9b59b6", "^",  9, "MultiHead M=3",          None),
        # ── 12E: 新追加ベースライン
        ("TT_item_partition",                     "#e74c3c", "h", 12, "Item Partition (Overlap=0)", None),
        # ── Plan 012 新手法
        ("TT_topk30_sj",                          "#2ecc71", "s", 11, "TopK-SoftJaccard (k=30)",None),
        ("TT_topk10_sj",                          "#27ae60", "s", 10, "TopK-SoftJaccard (k=10)",None),
        ("TT_adaptive_sj",                        "#e67e22", "P", 11, "Adaptive SoftJaccard",   None),
        ("TT_semantic_sj",                        "#1abc9c", "X", 11, "Semantic SoftJaccard",   None),
    ]

    claimed: set[str] = set()
    family_members: dict[str, list] = {}
    for prefix, *_ in FAMILIES:
        members = sorted(
            [(n, r) for n, r in all_results.items()
             if n not in claimed and (n == prefix or n.startswith(prefix + "_"))],
            key=lambda x: 1 - x[1]["mean"]["temporal_overlap"]
        )
        family_members[prefix] = members
        for n, _ in members:
            claimed.add(n)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8), facecolor="#0f1117")
    for ax in (ax0, ax1):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)

    for prefix, color, marker, size, label, _ in FAMILIES:
        members = family_members.get(prefix, [])
        if not members:
            continue
        divs = [1 - r["mean"]["temporal_overlap"] for _, r in members]
        rcs  = [r["mean"]["recall_cum"]            for _, r in members]
        ras  = [r["mean"]["recall_avg"]            for _, r in members]

        line_kw = dict(color=color, linewidth=1.2, alpha=0.5, zorder=1)
        if len(members) >= 2:
            order = np.argsort(divs)
            ax0.plot(np.array(divs)[order], np.array(rcs)[order], **line_kw)
            ax1.plot(np.array(divs)[order], np.array(ras)[order], **line_kw)

        for i, (name, r) in enumerate(members):
            lbl = label if i == 0 else "_nolegend_"
            div = 1 - r["mean"]["temporal_overlap"]
            rc  = r["mean"]["recall_cum"]
            ra  = r["mean"]["recall_avg"]
            rc_std = r["std"]["recall_cum"]
            ax0.scatter(div, rc, c=color, marker=marker, s=size**2,
                        alpha=0.92, label=lbl, edgecolors="white", linewidths=0.4, zorder=3)
            ax0.errorbar(div, rc, yerr=rc_std, fmt="none", color=color, alpha=0.4, zorder=2)
            ax1.scatter(div, ra, c=color, marker=marker, s=size**2,
                        alpha=0.92, label=lbl, edgecolors="white", linewidths=0.4, zorder=3)

            short = name.replace("TwoTower_d2_h64_", "").replace("TT_", "")
            short = short[:22]
            ax0.annotate(short, (div, rc), fontsize=5.5, color="#cccccc",
                         xytext=(3, 3), textcoords="offset points")
            ax1.annotate(short, (div, ra), fontsize=5.5, color="#cccccc",
                         xytext=(3, 3), textcoords="offset points")

    for ax, ylabel, title in [
        (ax0, "recall_cum  (N-trial cumulative Recall@10)", "Diversity vs recall_cum  [Plan 012]"),
        (ax1, "recall_avg  (per-trial mean Recall@10)",     "Diversity vs recall_avg  [Plan 012]"),
    ]:
        ax.set_xlabel("Diversity (1 - temporal_overlap)", color="#e0e0e0", fontsize=12)
        ax.set_ylabel(ylabel, color="#e0e0e0", fontsize=12)
        ax.set_title(title, color="#ffffff", fontsize=14, pad=12, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#555",
                  labelcolor="#e0e0e0", loc="best", framealpha=0.8)

    plt.tight_layout(pad=2.0)
    out_path = out_dir / "tradeoff_012_unified.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info(f"Saved plot -> {out_path}")


def print_summary(all_results: dict):
    log.info("\n" + "="*70 + "\nSummary\n" + "="*70)
    rows = sorted(
        [(n, r["mean"]["recall_cum"], r["mean"]["recall_avg"],
          r["mean"]["hit"], r["mean"]["temporal_overlap"])
         for n, r in all_results.items()],
        key=lambda x: -x[1]
    )
    header = f"{'model':<55} {'rc':>8} {'ra':>8} {'hit':>8} {'ov':>8}"
    log.info(header)
    for row in rows:
        log.info(f"{row[0]:<55} {row[1]:>8.4f} {row[2]:>8.4f} {row[3]:>8.4f} {row[4]:>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all",
                        choices=["all", "12A", "12B", "12C", "12E", "12D"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_all = args.subexp == "all"

    log.info("Loading MovieLens 1M ...")
    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(DATA_DIR)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt   = get_test_gt(interactions, uid2idx, iid2idx)
    n_total_items = len(item_embs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        log.info(f"Loaded {len(all_results)} existing results")
    else:
        all_results = {}

    baselines = load_baselines()
    for k, v in baselines.items():
        if k not in all_results:
            all_results[k] = v
    save_results(all_results, OUT_DIR)
    log.info(f"Loaded {len(baselines)} baselines from Plan 009/010/011")

    def rebuild_base_tt():
        tt = TwoTowerModel(
            hidden_dim=BEST_HIDDEN_DIM, depth=BEST_DEPTH,
            lr=1e-3, epochs=50, batch_size=1024,
            logit_scale=14.3, alpha=0.1,
        )
        log.info(f"Re-training base TwoTower_d2_h64 ...")
        tt.prepare(train_pos, user_embs, item_embs, device=args.device)
        return tt

    if run_all or args.subexp in ("12A", "12B", "12C", "12E"):
        best_tt = rebuild_base_tt()

        if run_all or args.subexp == "12E":
            r = run_12e(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

        if run_all or args.subexp == "12A":
            r = run_12a(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

        if run_all or args.subexp == "12B":
            r = run_12b(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

        if run_all or args.subexp == "12C":
            r = run_12c(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

    if run_all or args.subexp == "12D":
        plot_12d(all_results, OUT_DIR)

    print_summary(all_results)
    log.info("\n✅ Plan 012 completed!")


if __name__ == "__main__":
    main()
