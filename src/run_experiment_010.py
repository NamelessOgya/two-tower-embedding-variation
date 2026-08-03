"""
Plan 010 Experiment Runner
---------------------------
PostNoise vs soft_jaccard DivLoss の詳細比較分析

Sub-exp 10A: PostNoise σ 精密スイープ（diversity マッチ比較用）
Sub-exp 10B: K 感度分析 (K=5,10,20,50)
Sub-exp 10C: 余剰カバレッジの品質分析
Sub-exp 10D: 統計的有意差検定 (t 検定)
Sub-exp 10E: 統合プロット・レポート生成

Usage:
    PYTHONPATH=. python3 src/run_experiment_010.py --subexp all --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from scipy import stats

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.run_experiment import load_data, get_train_pos, get_test_gt, SEEDS, K, N_TRIALS
from src.evaluate.metrics import (
    recall_at_k, recall_at_k_single, hit_at_k, ndcg_at_k,
    temporal_overlap_rate,
)
from src.model.models_007 import TwoTowerModel, TwoTowerPostNoise
from src.model.models_008 import TwoTowerDivLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Parameters ────────────────────────────────────────────────────────────────
BEST_DEPTH      = 2
BEST_HIDDEN_DIM = 64
N_SEEDS         = len(SEEDS)

# 10A: PostNoise σ 精密スイープ
SIGMA_10A = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

# 10B: K感度分析
K_LIST = [5, 10, 20, 50]

OUT_DIR          = Path("report/plan_010")
DATA_DIR         = Path("data/processed/movielens")
PLAN_009_RESULTS = Path("report/plan_009/results.json")


# ── Evaluation Helpers ─────────────────────────────────────────────────────────

def evaluate_model_seed(
    model, test_gt, index, item_embs, k, n_trials, seed, n_total_items,
    return_per_user=False
):
    rng = np.random.default_rng(seed)
    per_user = defaultdict(list)
    user_trial_sets = {}  # user_idx -> list of sets (for extra coverage analysis)

    for user_idx, gt in test_gt.items():
        trial_lists = []
        trial_sets  = []
        for trial in range(n_trials):
            recs = model.recommend(user_idx, trial, rng, index, k)
            trial_lists.append(recs)
            trial_sets.append(set(recs))

        user_trial_sets[user_idx] = trial_sets

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
    if return_per_user:
        return mean_r, per_user, user_trial_sets
    return mean_r


def evaluate_model_k(model, test_gt, index, k, seed):
    """K感度分析用：指定 K で評価"""
    rng = np.random.default_rng(seed)
    per_user = defaultdict(list)
    for user_idx, gt in test_gt.items():
        trial_sets = []
        trial_lists = []
        for trial in range(N_TRIALS):
            recs = model.recommend(user_idx, trial, rng, index, k)
            trial_lists.append(recs)
            trial_sets.append(set(recs))
        per_user["recall_cum"].append(recall_at_k(trial_sets, gt))
        per_user["recall_avg"].append(
            float(np.mean([recall_at_k_single(s, gt) for s in trial_sets]))
        )
        per_user["temporal_overlap"].append(temporal_overlap_rate(trial_sets, k))
    return {k2: float(np.mean(v)) for k2, v in per_user.items()}


def run_model_eval(model, test_gt, train_pos, user_embs, item_embs, device, n_total_items):
    model.prepare(train_pos, user_embs, item_embs, device=device)
    index = model.build_index()
    seed_results = []
    for seed in range(N_SEEDS):
        r = evaluate_model_seed(
            model, test_gt, index, item_embs, K, N_TRIALS, seed, n_total_items
        )
        log.info(f"  seed={seed}  rc={r['recall_cum']:.4f}  ra={r['recall_avg']:.4f}  "
                 f"hit={r['hit']:.4f}  ov={r['temporal_overlap']:.4f}")
        seed_results.append(r)
    keys = seed_results[0].keys()
    mean = {k2: float(np.mean([r[k2] for r in seed_results])) for k2 in keys}
    std  = {k2: float(np.std( [r[k2] for r in seed_results])) for k2 in keys}
    raw  = {k2: [r[k2] for r in seed_results] for k2 in keys}
    log.info(f"  [AVG] rc={mean['recall_cum']:.4f}±{std['recall_cum']:.4f}  "
             f"ra={mean['recall_avg']:.4f}±{std['recall_avg']:.4f}  "
             f"ov={mean['temporal_overlap']:.4f}±{std['temporal_overlap']:.4f}")
    return {"mean": mean, "std": std, "raw": raw}


def save_results(all_results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)


# ── Sub-exp 10A: PostNoise σ 精密スイープ ────────────────────────────────────

def run_10a(base_tt, train_pos, test_gt, user_embs, item_embs, device, n_total_items):
    log.info("\n" + "="*60 + "\nSub-exp 10A: PostNoise sigma fine sweep\n" + "="*60)
    results = {}
    for sigma in SIGMA_10A:
        model = TwoTowerPostNoise(base_tt=base_tt, sigma=sigma)
        log.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
        results[model.name] = run_model_eval(
            model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
    return results


# ── Sub-exp 10B: K 感度分析 ──────────────────────────────────────────────────

def run_10b(base_tt, train_pos, test_gt, user_embs, item_embs, device, n_total_items, all_results):
    log.info("\n" + "="*60 + "\nSub-exp 10B: K sensitivity analysis\n" + "="*60)

    # 比較対象モデル
    # PostNoise σ=0.10 (010A), soft_jaccard λ=0.1 (009 results)
    postnoise_s10_name = f"TwoTower_d{BEST_DEPTH}_h{BEST_HIDDEN_DIM}_postnoise_s0p1"
    model_pn  = TwoTowerPostNoise(base_tt=base_tt, sigma=0.10)
    model_pn.prepare(train_pos, user_embs, item_embs, device=device)
    index_pn  = model_pn.build_index()

    sj_name = "TT_divloss_soft_jaccard_l0p1_s0p05"
    model_sj  = TwoTowerDivLoss(
        base_tt=base_tt, div_loss_name="soft_jaccard",
        lambda_div=0.1, sigma=0.05, lr=2e-3, epochs=30, batch_size=512,
    )
    model_sj.prepare(train_pos, user_embs, item_embs, device=device)
    index_sj = model_sj.build_index()

    k_rows = []
    for k_val in K_LIST:
        for model_name, model, idx in [
            (postnoise_s10_name, model_pn, index_pn),
            (sj_name,            model_sj, index_sj),
        ]:
            seed_rcs, seed_ras = [], []
            for seed in range(N_SEEDS):
                r = evaluate_model_k(model, test_gt, idx, k_val, seed)
                seed_rcs.append(r["recall_cum"])
                seed_ras.append(r["recall_avg"])
            k_rows.append({
                "model": model_name, "K": k_val,
                "recall_cum_mean": float(np.mean(seed_rcs)),
                "recall_cum_std":  float(np.std(seed_rcs)),
                "recall_avg_mean": float(np.mean(seed_ras)),
                "recall_avg_std":  float(np.std(seed_ras)),
            })
            log.info(f"  {model_name}  K={k_val}  "
                     f"rc={np.mean(seed_rcs):.4f}±{np.std(seed_rcs):.4f}  "
                     f"ra={np.mean(seed_ras):.4f}±{np.std(seed_ras):.4f}")

    df = pd.DataFrame(k_rows)
    df.to_csv(OUT_DIR / "k_sensitivity.csv", index=False)
    log.info(f"Saved K sensitivity -> {OUT_DIR / 'k_sensitivity.csv'}")
    return df


# ── Sub-exp 10C: 余剰カバレッジの品質分析 ────────────────────────────────────

def run_10c(base_tt, train_pos, test_gt, user_embs, item_embs, device):
    log.info("\n" + "="*60 + "\nSub-exp 10C: Extra coverage quality analysis\n" + "="*60)

    # σ=0.10 と λ=0.1 で diversity が近い2点を比較
    model_pn = TwoTowerPostNoise(base_tt=base_tt, sigma=0.10)
    model_pn.prepare(train_pos, user_embs, item_embs, device=device)
    index_pn = model_pn.build_index()

    model_sj = TwoTowerDivLoss(
        base_tt=base_tt, div_loss_name="soft_jaccard",
        lambda_div=0.1, sigma=0.05, lr=2e-3, epochs=30, batch_size=512,
    )
    model_sj.prepare(train_pos, user_embs, item_embs, device=device)
    index_sj = model_sj.build_index()

    rng = np.random.default_rng(42)
    rows = []
    n_users_analyzed = 0

    for user_idx, gt in test_gt.items():
        gt_set = gt  # すでに set

        pn_sets = [set(model_pn.recommend(user_idx, t, rng, index_pn, K)) for t in range(N_TRIALS)]
        rng2 = np.random.default_rng(42)
        sj_sets = [set(model_sj.recommend(user_idx, t, rng2, index_sj, K)) for t in range(N_TRIALS)]

        pn_all = set().union(*pn_sets)
        sj_all = set().union(*sj_sets)

        # soft_jaccard が独自にカバーするアイテム（PostNoise にないもの）
        sj_extra = sj_all - pn_all
        pn_extra = pn_all - sj_all

        # extra の中で GT に含まれるものの割合
        sj_extra_prec = len(sj_extra & gt_set) / max(len(sj_extra), 1)
        pn_extra_prec = len(pn_extra & gt_set) / max(len(pn_extra), 1)

        rows.append({
            "user_idx":      user_idx,
            "pn_coverage":   len(pn_all),
            "sj_coverage":   len(sj_all),
            "sj_extra_count":len(sj_extra),
            "pn_extra_count":len(pn_extra),
            "sj_extra_prec": sj_extra_prec,
            "pn_extra_prec": pn_extra_prec,
            "gt_size":       len(gt_set),
        })
        n_users_analyzed += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "extra_coverage.csv", index=False)

    log.info(f"\n--- Extra Coverage Quality (N={n_users_analyzed} users) ---")
    log.info(f"  SJ mean coverage:    {df['sj_coverage'].mean():.1f} items")
    log.info(f"  PN mean coverage:    {df['pn_coverage'].mean():.1f} items")
    log.info(f"  SJ extra vs PN (count): {df['sj_extra_count'].mean():.1f}")
    log.info(f"  PN extra vs SJ (count): {df['pn_extra_count'].mean():.1f}")
    log.info(f"  SJ extra precision (GT hit rate): {df['sj_extra_prec'].mean():.4f}")
    log.info(f"  PN extra precision (GT hit rate): {df['pn_extra_prec'].mean():.4f}")
    return df


# ── Sub-exp 10D: 統計的有意差検定 ────────────────────────────────────────────

def run_10d(all_results: dict):
    log.info("\n" + "="*60 + "\nSub-exp 10D: Statistical significance test\n" + "="*60)

    pn_name = "TwoTower_d2_h64_postnoise_s0p1"
    sj_name = "TT_divloss_soft_jaccard_l0p1_s0p05"

    rows = []
    for name_a, name_b in [(pn_name, sj_name)]:
        for metric in ["recall_cum", "recall_avg", "hit"]:
            if name_a not in all_results or name_b not in all_results:
                continue
            if "raw" not in all_results[name_a] or "raw" not in all_results[name_b]:
                log.warning(f"No raw data for {name_a} or {name_b}")
                continue
            a_vals = all_results[name_a]["raw"][metric]
            b_vals = all_results[name_b]["raw"][metric]
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(a_vals, b_vals)
            winner = name_b if np.mean(b_vals) > np.mean(a_vals) else name_a
            rows.append({
                "metric": metric,
                "PostNoise_mean": np.mean(a_vals),
                "SJ_mean":        np.mean(b_vals),
                "t_stat":  t_stat,
                "p_value": p_val,
                "significant_p05": p_val < 0.05,
                "winner":  winner,
            })
            log.info(f"  [{metric}] PN={np.mean(a_vals):.4f}  SJ={np.mean(b_vals):.4f}  "
                     f"t={t_stat:.3f}  p={p_val:.4f}  sig={'Yes' if p_val < 0.05 else 'No'}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "significance_test.csv", index=False)
    log.info(f"Saved significance test -> {OUT_DIR / 'significance_test.csv'}")
    return df


# ── Sub-exp 10E: 統合プロット ─────────────────────────────────────────────────

def plot_10e(all_results: dict, k_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor="#0f1117")

    # --- Panel 1: diversity vs recall_cum (PostNoise sweep + SJ sweep) ---
    ax = axes[0]
    ax.set_facecolor("#1a1d27")
    ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax.tick_params(colors="#e0e0e0")
    for spine in ax.spines.values():
        spine.set_color("#2d3142")

    # PostNoise 全σ
    pn_points = sorted(
        [(r["mean"]["temporal_overlap"], r["mean"]["recall_cum"], name)
         for name, r in all_results.items() if "postnoise" in name],
        key=lambda x: x[0]
    )
    if pn_points:
        pn_divs = [1 - x[0] for x in pn_points]
        pn_rcs  = [x[1] for x in pn_points]
        ax.plot(pn_divs, pn_rcs, color="#e74c3c", linewidth=1.5, alpha=0.7, label="PostNoise (σ sweep)")
        for div, rc, name in zip(pn_divs, pn_rcs, [x[2] for x in pn_points]):
            sigma_str = name.split("_s")[-1].replace("p", ".")
            ax.scatter(div, rc, c="#e74c3c", marker="v", s=81, alpha=0.9, zorder=3)
            ax.annotate(f"σ={sigma_str}", (div, rc), fontsize=6, color="#dddddd",
                        xytext=(3, 3), textcoords="offset points")

    # soft_jaccard fine sweep from plan_009
    sj_points = sorted(
        [(r["mean"]["temporal_overlap"], r["mean"]["recall_cum"], name)
         for name, r in all_results.items()
         if "soft_jaccard" in name and "postnoise" not in name],
        key=lambda x: x[0]
    )
    if sj_points:
        sj_divs = [1 - x[0] for x in sj_points]
        sj_rcs  = [x[1] for x in sj_points]
        ax.plot(sj_divs, sj_rcs, color="#f1c40f", linewidth=1.5, alpha=0.7, label="SoftJaccard (λ sweep)")
        for div, rc, name in zip(sj_divs, sj_rcs, [x[2] for x in sj_points]):
            lam_str = name.split("_l")[-1].split("_")[0].replace("p", ".")
            ax.scatter(div, rc, c="#f1c40f", marker="*", s=100, alpha=0.9, zorder=3)
            ax.annotate(f"λ={lam_str}", (div, rc), fontsize=6, color="#dddddd",
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Diversity (1 - temporal_overlap)", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("recall_cum", color="#e0e0e0", fontsize=11)
    ax.set_title("Panel 1: Diversity vs recall_cum", color="#ffffff", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#555", labelcolor="#e0e0e0")

    # --- Panel 2: K sensitivity ---
    ax2 = axes[1]
    ax2.set_facecolor("#1a1d27")
    ax2.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax2.tick_params(colors="#e0e0e0")
    for spine in ax2.spines.values():
        spine.set_color("#2d3142")

    if k_df is not None and not k_df.empty:
        for model_name, color, label in [
            ("postnoise", "#e74c3c", "PostNoise σ=0.10"),
            ("soft_jaccard", "#f1c40f", "SoftJaccard λ=0.1"),
        ]:
            sub = k_df[k_df["model"].str.contains(model_name)].sort_values("K")
            if not sub.empty:
                ax2.plot(sub["K"], sub["recall_cum_mean"], color=color,
                         marker="o", linewidth=1.5, label=label)
                ax2.fill_between(
                    sub["K"],
                    sub["recall_cum_mean"] - sub["recall_cum_std"],
                    sub["recall_cum_mean"] + sub["recall_cum_std"],
                    color=color, alpha=0.15
                )

    ax2.set_xlabel("K (list length)", color="#e0e0e0", fontsize=11)
    ax2.set_ylabel("recall_cum", color="#e0e0e0", fontsize=11)
    ax2.set_title("Panel 2: K Sensitivity", color="#ffffff", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#555", labelcolor="#e0e0e0")

    # --- Panel 3: recall_avg comparison ---
    ax3 = axes[2]
    ax3.set_facecolor("#1a1d27")
    ax3.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)
    ax3.tick_params(colors="#e0e0e0")
    for spine in ax3.spines.values():
        spine.set_color("#2d3142")

    if pn_points:
        pn_ras = [all_results[x[2]]["mean"]["recall_avg"] for x in pn_points]
        ax3.plot(pn_divs, pn_ras, color="#e74c3c", linewidth=1.5, alpha=0.7,
                 marker="v", label="PostNoise (σ sweep)")
    if sj_points:
        sj_ras = [all_results[x[2]]["mean"]["recall_avg"] for x in sj_points]
        ax3.plot(sj_divs, sj_ras, color="#f1c40f", linewidth=1.5, alpha=0.7,
                 marker="*", label="SoftJaccard (λ sweep)")

    ax3.set_xlabel("Diversity (1 - temporal_overlap)", color="#e0e0e0", fontsize=11)
    ax3.set_ylabel("recall_avg", color="#e0e0e0", fontsize=11)
    ax3.set_title("Panel 3: Diversity vs recall_avg", color="#ffffff", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#555", labelcolor="#e0e0e0")

    plt.tight_layout(pad=2.0)
    out_path = out_dir / "comparison_010.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info(f"Saved comparison plot -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all",
                        choices=["all", "10A", "10B", "10C", "10D", "10E"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_all = args.subexp == "all"

    log.info("Loading MovieLens 1M ...")
    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(DATA_DIR)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt   = get_test_gt(interactions, uid2idx, iid2idx)
    n_total_items = len(item_embs)
    log.info(f"  users={len(train_pos)}, items={n_total_items}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 既存結果ロード
    results_path = OUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        log.info(f"Loaded {len(all_results)} existing results")
    else:
        all_results = {}

    # Plan 009 結果をベースラインとして取り込む（soft_jaccard sweep + postnoise s0p05/s0p1）
    if PLAN_009_RESULTS.exists():
        with open(PLAN_009_RESULTS) as f:
            r009 = json.load(f)
        baseline_keys = [
            k for k in r009
            if "soft_jaccard" in k or k in [
                "TwoTower_d2_h64",
                "TwoTower_d2_h64_postnoise_s0p05",
                "TwoTower_d2_h64_postnoise_s0p1",
            ]
        ]
        for k in baseline_keys:
            if k in r009 and k not in all_results:
                all_results[k] = r009[k]
        save_results(all_results, OUT_DIR)
        log.info(f"Loaded {len(baseline_keys)} baselines from Plan 009")

    # base TT 再学習（10A/10B/10C で共有）
    def rebuild_best_tt():
        tt = TwoTowerModel(
            hidden_dim=BEST_HIDDEN_DIM, depth=BEST_DEPTH,
            lr=1e-3, epochs=50, batch_size=1024,
            logit_scale=14.3, alpha=0.1,
        )
        log.info(f"Re-training TwoTower_d{BEST_DEPTH}_h{BEST_HIDDEN_DIM} ...")
        tt.prepare(train_pos, user_embs, item_embs, device=args.device)
        return tt

    k_df = None

    if run_all or args.subexp in ("10A", "10B", "10C"):
        best_tt = rebuild_best_tt()

        if run_all or args.subexp == "10A":
            r = run_10a(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

        if run_all or args.subexp == "10B":
            k_df = run_10b(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items, all_results)

        if run_all or args.subexp == "10C":
            cov_df = run_10c(best_tt, train_pos, test_gt, user_embs, item_embs, args.device)

    if run_all or args.subexp == "10D":
        sig_df = run_10d(all_results)

    if run_all or args.subexp == "10E":
        if k_df is None and (OUT_DIR / "k_sensitivity.csv").exists():
            k_df = pd.read_csv(OUT_DIR / "k_sensitivity.csv")
        plot_10e(all_results, k_df, OUT_DIR)

    log.info("\n✅ Plan 010 completed!")


if __name__ == "__main__":
    main()
