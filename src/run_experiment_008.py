"""
Plan 008 実験ランナー
---------------------
Two-Tower MLP (Plan 007 最良モデル) をベースに、
多様性制御手法を強化した追加実験を実施する。

Sub-exp:
    8A: 参照ベースライン（Plan 007 の results.json をロードして参照）
    8B: Two-Tower + Diversity Loss 統合学習 (6損失 × 3λ = 18条件)
    8C: TrainNoise 位置バリアント × 推論時ノイズあり (2種 × 4σ = 8条件)
    8D: 統合 Pareto プロット
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
import torch

from src.run_experiment import (
    load_data, get_train_pos, get_test_gt,
    SEEDS, K, N_TRIALS,
)
from src.evaluate.metrics import (
    recall_at_k, recall_at_k_single, hit_at_k, ndcg_at_k,
    temporal_overlap_rate, intra_list_diversity, coverage,
)
from src.model.models_005 import M0_EnhancedBase
from src.model.models_007 import TwoTowerModel, TwoTowerPostNoise
from src.model.models_008 import TwoTowerDivLoss, TwoTowerInputNoiseBoth, TwoTowerOutputNoiseBoth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Sweep Parameters ───────────────────────────────────────────────────────────
BEST_DEPTH      = 2
BEST_HIDDEN_DIM = 64

DIV_LOSS_NAMES = ["cosine_emb", "l2_emb", "kl_dist", "js_dist", "soft_jaccard", "listnet"]
LAMBDA_SWEEP   = [0.1, 0.5, 1.0]   # 6 losses × 3λ = 18 条件
SIGMA_DIV      = 0.05               # 8B の推論ノイズ σ（7D の最良値）

SIGMA_NOISE_BOTH = [0.01, 0.02, 0.05, 0.1]  # 8C: 2バリアント × 4σ = 8条件

N_SEEDS = len(SEEDS)   # src.run_experiment から import した SEEDS リストの長さ

OUT_DIR  = Path("report/plan_008")
DATA_DIR = Path("data/processed/movielens")
PLAN_007_RESULTS = Path("report/plan_007/results.json")


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
    """N_SEEDS 回評価して mean/std を返す。"""
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


# ── Sub-exp 8A: ベースライン参照（Plan 007 から） ────────────────────────────

def load_007_baselines(plan007_path: Path) -> dict:
    """Plan 007 の結果から参照ベースラインをロードする。"""
    if not plan007_path.exists():
        log.warning(f"Plan 007 results not found at {plan007_path}. Skipping 8A.")
        return {}
    with open(plan007_path) as f:
        r007 = json.load(f)
    # 比較用ベースラインのみ抽出
    baseline_keys = [
        "M0_strong",
        "TwoTower_d2_h64",
        "TwoTower_d2_h64_postnoise_s0p05",  # メインベースライン
        "TwoTower_d2_h64_postnoise_s0p1",
        "TT_soft_jaccard_l0p5",
    ]
    return {k: r007[k] for k in baseline_keys if k in r007}


# ── Sub-exp 8B: Two-Tower + Diversity Loss ────────────────────────────────────

def run_8b(base_tt: TwoTowerModel, train_pos, test_gt, user_embs, item_embs, device, n_total_items) -> dict:
    log.info("\n" + "="*60 + "\nSub-exp 8B: Two-Tower + Diversity Loss (joint BPR+div)\n" + "="*60)
    results = {}
    for loss_name in DIV_LOSS_NAMES:
        for lam in LAMBDA_SWEEP:
            model = TwoTowerDivLoss(
                base_tt=base_tt,
                div_loss_name=loss_name,
                lambda_div=lam,
                sigma=SIGMA_DIV,
                lr=2e-3,
                epochs=30,
                batch_size=512,
            )
            results[model.name] = run_model_eval(
                model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
            )
    return results


# ── Sub-exp 8C: TrainNoise 位置バリアント × 推論時ノイズあり ─────────────────

def run_8c(best_depth: int, best_hidden_dim: int, train_pos, test_gt, user_embs, item_embs, device, n_total_items) -> dict:
    log.info("\n" + "="*60 + "\nSub-exp 8C: TrainNoise variants (input/output x train+infer)\n" + "="*60)
    results = {}
    for sigma in SIGMA_NOISE_BOTH:
        # Input noise: whitened 入力にノイズ（学習時・推論時）
        m_in = TwoTowerInputNoiseBoth(
            hidden_dim=best_hidden_dim, depth=best_depth,
            sigma=sigma, lr=1e-3, epochs=50, batch_size=1024,
        )
        results[m_in.name] = run_model_eval(
            m_in, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
        # Output noise: MLP 出力にノイズ（学習時・推論時）
        m_out = TwoTowerOutputNoiseBoth(
            hidden_dim=best_hidden_dim, depth=best_depth,
            sigma=sigma, lr=1e-3, epochs=50, batch_size=1024,
        )
        results[m_out.name] = run_model_eval(
            m_out, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
    return results


# ── Sub-exp 8D: Pareto Frontier 統合プロット ──────────────────────────────────

def plot_unified_008(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── ファミリー定義（同一グループのモデルを線で結ぶ） ─────────────────────
    # (prefix, color, marker, size, label, param_key)
    # param_key: 点のアノテーションに使うパラメータ名（λ or σ）
    FAMILIES = [
        # ベースライン（線なし・単点）
        ("M0_strong",              "#aaaaaa", "o", 11, "M0_strong",           None),
        ("TwoTower_d2_h64",        "#3498db", "s", 10, "TT_d2_h64 (no div)", None),
        # PostNoise（σ でソート → 線で結ぶ）
        ("TwoTower_d2_h64_postnoise", "#e74c3c", "v", 9, "PostNoise (σ sweep)", "σ"),
        # soft_jaccard (007C, λ=0.5 単点)
        ("TT_soft_jaccard",        "#9b59b6", "^", 9, "TT+soft_jaccard (007C)", None),
        # 8B: DivLoss 各ファミリー
        ("TT_divloss_cosine_emb",  "#2ecc71", "P", 9, "DivLoss cosine_emb",  "λ"),
        ("TT_divloss_l2_emb",      "#27ae60", "D", 9, "DivLoss l2_emb",      "λ"),
        ("TT_divloss_kl_dist",     "#95a5a6", "x", 8, "DivLoss kl_dist",     "λ"),
        ("TT_divloss_js_dist",     "#1abc9c", "h", 9, "DivLoss js_dist",     "λ"),
        ("TT_divloss_soft_jaccard","#f39c12", "*",11, "DivLoss soft_jaccard","λ"),
        ("TT_divloss_listnet",     "#8e44ad", "p", 9, "DivLoss listnet",     "λ"),
        # 8C: Noise both
        ("TT_inputnoise_both",     "#e67e22", "<", 9, "InputNoise Both",     "σ"),
        ("TT_outputnoise_both",    "#d35400", ">", 9, "OutputNoise Both",    "σ"),
    ]

    # param label mapping (λ/σ 値を末尾から取り出す)
    LAMBDA_MAP = {"l0p1": "λ=0.1", "l0p5": "λ=0.5", "l1p0": "λ=1.0"}
    SIGMA_MAP  = {
        "s0p01": "σ=0.01", "s0p02": "σ=0.02", "s0p05": "σ=0.05",
        "s0p1": "σ=0.10", "s0p5": "σ=0.50",
        "postnoise_s0p05": "σ=0.05", "postnoise_s0p1": "σ=0.10",
    }

    def get_param_label(name: str) -> str:
        for k, v in {**LAMBDA_MAP, **SIGMA_MAP}.items():
            if name.endswith(k) or ("_" + k + "_") in name:
                return v
        # postnoise case
        for k, v in SIGMA_MAP.items():
            if k in name:
                return v
        return name.split("_")[-1]

    # ── データ収集 ────────────────────────────────────────────────────────────
    data: dict[str, dict] = {}
    for name, r in all_results.items():
        m = r["mean"]
        data[name] = {
            "div":  1.0 - m["temporal_overlap"],
            "rc":   m["recall_cum"],
            "ra":   m["recall_avg"],
            "hit":  m["hit"],
        }

    # ── 描画 ─────────────────────────────────────────────────────────────────
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8), facecolor="#0f1117")
    for ax in (ax0, ax1):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.6)

    for prefix, color, marker, size, label, param_key in FAMILIES:
        # このファミリーに属するモデルを抽出し diversity 順にソート
        members = sorted(
            [(n, d) for n, d in data.items() if n == prefix or n.startswith(prefix + "_")],
            key=lambda x: x[1]["div"]
        )
        if not members:
            continue

        divs = [m[1]["div"] for m in members]
        rcs  = [m[1]["rc"]  for m in members]
        ras  = [m[1]["ra"]  for m in members]

        # 2点以上あれば線で結ぶ
        line_kw = dict(color=color, linewidth=1.2, alpha=0.5, zorder=1)
        if len(members) >= 2:
            ax0.plot(divs, rcs, **line_kw)
            ax1.plot(divs, ras, **line_kw)

        # 散布点とアノテーション
        for i, (name, d) in enumerate(members):
            lbl = label if i == 0 else "_nolegend_"
            ax0.scatter(d["div"], d["rc"], c=color, marker=marker, s=size**2,
                        alpha=0.9, label=lbl, edgecolors="white", linewidths=0.4, zorder=2)
            ax1.scatter(d["div"], d["ra"], c=color, marker=marker, s=size**2,
                        alpha=0.9, label=lbl, edgecolors="white", linewidths=0.4, zorder=2)
            ann = get_param_label(name)
            ax0.annotate(ann, (d["div"], d["rc"]),
                         fontsize=6.5, color="#dddddd",
                         xytext=(4, 4), textcoords="offset points")
            ax1.annotate(ann, (d["div"], d["ra"]),
                         fontsize=6.5, color="#dddddd",
                         xytext=(4, 4), textcoords="offset points")

    xlabel = "Diversity (1 - temporal_overlap)"
    for ax, ylabel, title in [
        (ax0, "recall_cum  (N-trial cumulative Recall)",  "Diversity vs recall_cum  [Plan 008]"),
        (ax1, "recall_avg  (per-trial mean Recall)",      "Diversity vs recall_avg  [Plan 008]"),
    ]:
        ax.set_xlabel(xlabel, color="#e0e0e0", fontsize=12)
        ax.set_ylabel(ylabel, color="#e0e0e0", fontsize=12)
        ax.set_title(title, color="#ffffff", fontsize=14, pad=12, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#555",
                  labelcolor="#e0e0e0", loc="best", framealpha=0.8)

    plt.tight_layout(pad=2.0)
    out_path = out_dir / "tradeoff_008_unified.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info(f"Saved unified tradeoff plot → {out_path}")


# ── 結果サマリー表示 ──────────────────────────────────────────────────────────

def print_summary(all_results: dict):
    rows = []
    for name, r in all_results.items():
        m, s = r["mean"], r["std"]
        rows.append({
            "model":            name,
            "recall_cum":       f"{m['recall_cum']:.4f}±{s['recall_cum']:.4f}",
            "recall_avg":       f"{m['recall_avg']:.4f}±{s['recall_avg']:.4f}",
            "hit":              f"{m['hit']:.4f}±{s['hit']:.4f}",
            "ndcg":             f"{m['ndcg']:.4f}±{s['ndcg']:.4f}",
            "temporal_overlap": f"{m['temporal_overlap']:.4f}±{s['temporal_overlap']:.4f}",
        })
    df = pd.DataFrame(rows)
    log.info("\n" + df.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all",
                        choices=["all", "8B", "8C", "8D"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_all = args.subexp == "all"

    # データロード
    log.info("Loading MovieLens 1M ...")
    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(DATA_DIR)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt   = get_test_gt(interactions, uid2idx, iid2idx)
    n_total_items = len(item_embs)
    log.info(f"  users={len(train_pos)}, items={n_total_items}, "
             f"user_emb={user_embs.shape}, item_emb={item_embs.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 既存結果のロード（再開対応）
    results_path = OUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        log.info(f"Loaded {len(all_results)} existing results from {results_path}")
    else:
        all_results = {}

    # 8A: Plan 007 ベースライン参照
    if run_all:
        baselines = load_007_baselines(PLAN_007_RESULTS)
        all_results.update(baselines)
        log.info(f"Loaded {len(baselines)} baselines from Plan 007: {list(baselines.keys())}")
        save_results(all_results, OUT_DIR)

    # ベスト Two-Tower を再学習（8B/8C のベースとして使用）
    def rebuild_best_tt() -> TwoTowerModel:
        """Plan 007 の最良 Two-Tower を再学習して返す。"""
        tt = TwoTowerModel(
            hidden_dim=BEST_HIDDEN_DIM,
            depth=BEST_DEPTH,
            lr=1e-3,
            epochs=50,
            batch_size=1024,
            logit_scale=14.3,
            alpha=0.1,
        )
        log.info(f"Re-training best Two-Tower (depth={BEST_DEPTH}, hidden_dim={BEST_HIDDEN_DIM}) ...")
        tt.prepare(train_pos, user_embs, item_embs, device=args.device)
        return tt

    # 8B: Diversity Loss 統合学習
    if run_all or args.subexp == "8B":
        best_tt = rebuild_best_tt()
        r = run_8b(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
        all_results.update(r)
        save_results(all_results, OUT_DIR)

    # 8C: TrainNoise 位置バリアント × 推論時ノイズあり
    if run_all or args.subexp == "8C":
        r = run_8c(
            BEST_DEPTH, BEST_HIDDEN_DIM,
            train_pos, test_gt, user_embs, item_embs, args.device, n_total_items
        )
        all_results.update(r)
        save_results(all_results, OUT_DIR)

    # 8D: 統合 Pareto プロット
    if run_all or args.subexp == "8D":
        print_summary(all_results)
        plot_unified_008(all_results, OUT_DIR)

    log.info("\n✅ Plan 008 completed!")


if __name__ == "__main__":
    main()
