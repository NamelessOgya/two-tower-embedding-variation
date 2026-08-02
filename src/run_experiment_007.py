"""
Plan 007 Experiment Runner: Two-Tower MLP Head
-----------------------------------------------
Sub-exp 7A: M0_strong ベースライン（比較用）
Sub-exp 7B: Two-Tower MLP スイープ
             depth ∈ {1, 2, 3} × hidden_dim ∈ {64, 128, 256} = 9条件
Sub-exp 7C: 最良 Two-Tower + soft_jaccard アダプタ (λ スイープ) = 5条件
Sub-exp 7D: 最良 Two-Tower + 推論時ノイズ (Post-Noise, σ スイープ) = 6条件
Sub-exp 7E: Two-Tower 学習時ノイズ (Train-Noise, σ スイープ, 最良 depth/dim) = 6条件
Sub-exp 7F: 統合 Pareto Frontier プロット生成

Usage:
    PYTHONPATH=. python3 src/run_experiment_007.py --subexp all --device cuda
    PYTHONPATH=. python3 src/run_experiment_007.py --subexp 7B --device cuda
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
import torch
import torch.nn.functional as F

from src.run_experiment import (
    load_data, get_train_pos, get_test_gt,
    SEEDS, K, N_TRIALS,
)
from src.evaluate.metrics import (
    recall_at_k, recall_at_k_single, hit_at_k, ndcg_at_k,
    temporal_overlap_rate, intra_list_diversity, coverage,
)
from src.model.models_005 import M0_EnhancedBase
from src.model.models_007 import TwoTowerModel, TwoTowerPostNoise, TwoTowerTrainNoise

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Sweep Parameters ───────────────────────────────────────────────────────────
DEPTHS       = [1, 2, 3]
HIDDEN_DIMS  = [64, 128, 256]
LAMBDA_SWEEP = [0.01, 0.05, 0.1, 0.5, 1.0]
SIGMA_SWEEP  = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

OUT_DIR  = Path("report/plan_007")
DATA_DIR = Path("data/processed/movielens")


# ── Evaluation Helpers ─────────────────────────────────────────────────────────

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


def run_model_eval(model, test_gt, train_pos, user_embs, item_embs, device, n_total_items):
    """1モデルを全シードで評価して結果を返す。"""
    log.info(f"\n{'='*60}\nModel: {model.name}\n{'='*60}")
    t0 = time.time()
    model.prepare(train_pos, user_embs, item_embs, device=device)
    log.info(f"  prepare: {time.time()-t0:.1f}s")

    # TwoTowerModel は専用の FAISS インデックスを保持
    if isinstance(model, TwoTowerModel):
        index = model.build_index()
        eval_item_embs = model._proj_item_embs
    elif hasattr(model, "transformed_item_embs") and model.transformed_item_embs is not None:
        idx = faiss.IndexFlatIP(model.transformed_item_embs.shape[1])
        idx.add(model.transformed_item_embs.astype(np.float32))
        index = idx
        eval_item_embs = model.transformed_item_embs
    else:
        idx = faiss.IndexFlatIP(item_embs.shape[1])
        idx.add(item_embs.astype(np.float32))
        index = idx
        eval_item_embs = item_embs

    seed_results = []
    for seed in SEEDS:
        m = evaluate_model_seed(
            model, test_gt, index, eval_item_embs,
            k=K, n_trials=N_TRIALS, seed=seed,
            n_total_items=n_total_items,
        )
        log.info(
            f"  seed={seed}  rc={m['recall_cum']:.4f}  ra={m['recall_avg']:.4f}  "
            f"hit={m['hit']:.4f}  ov={m['temporal_overlap']:.4f}  cov={m['coverage']:.4f}"
        )
        seed_results.append(m)

    avg = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
    std = {k: float(np.std([r[k] for r in seed_results]))  for k in seed_results[0]}
    log.info(
        f"  [AVG] rc={avg['recall_cum']:.4f}±{std['recall_cum']:.4f}  "
        f"ra={avg['recall_avg']:.4f}±{std['recall_avg']:.4f}  "
        f"hit={avg['hit']:.4f}±{std['hit']:.4f}  "
        f"ov={avg['temporal_overlap']:.4f}±{std['temporal_overlap']:.4f}"
    )
    return {"mean": avg, "std": std, "per_seed": seed_results}


def save_results(results: dict, out_dir: Path, filename: str = "results.json"):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename, "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for name, r in results.items():
        row = {"model": name}
        for k in r["mean"]:
            row[k] = f"{r['mean'][k]:.4f}±{r['std'][k]:.4f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    log.info(f"\n{df[['model','recall_cum','recall_avg','hit','ndcg','temporal_overlap']].to_string(index=False)}")


# ── Sub-exp 7A: M0_strong Baseline ────────────────────────────────────────────

def run_7a(train_pos, test_gt, user_embs, item_embs, device, n_total_items):
    log.info("\n" + "="*60 + "\nSub-exp 7A: M0_strong Baseline\n" + "="*60)
    model = M0_EnhancedBase(
        name="M0_strong",
        use_whitening=True,
        use_logq=True,
        logit_scale=14.3,
        alpha=0.1,
    )
    result = run_model_eval(model, test_gt, train_pos, user_embs, item_embs, device, n_total_items)
    return {"M0_strong": result}


# ── Sub-exp 7B: Two-Tower MLP Sweep ───────────────────────────────────────────

def run_7b(train_pos, test_gt, user_embs, item_embs, device, n_total_items):
    log.info("\n" + "="*60 + "\nSub-exp 7B: Two-Tower MLP Sweep\n" + "="*60)
    results = {}
    for depth in DEPTHS:
        for hidden_dim in HIDDEN_DIMS:
            model = TwoTowerModel(
                hidden_dim=hidden_dim,
                depth=depth,
                lr=1e-3,
                epochs=50,
                batch_size=1024,
                logit_scale=14.3,
                alpha=0.1,
            )
            results[model.name] = run_model_eval(
                model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
            )
    return results


# ── Sub-exp 7C: Best Two-Tower + soft_jaccard Adapter ─────────────────────────

def _find_best_tt(results_7b: dict) -> tuple[int, int]:
    """7B の結果から Hit@10 が最良の depth, hidden_dim を返す。"""
    best_name, best_hit = None, -1.0
    for name, r in results_7b.items():
        hit = r["mean"]["hit"]
        if hit > best_hit:
            best_hit = hit
            best_name = name
    # "TwoTower_d{depth}_h{hidden_dim}" からパース
    parts = best_name.split("_")
    depth = int(parts[1][1:])
    hidden_dim = int(parts[2][1:])
    log.info(f"Best Two-Tower: {best_name} (hit={best_hit:.4f}) → depth={depth}, hidden_dim={hidden_dim}")
    return depth, hidden_dim


class _TwoTowerWithSoftJaccard(TwoTowerModel):
    """
    Two-Tower + soft_jaccard diversity adapter（ユーザータワーにノイズを追加）。
    学習済み TwoTower の重みを引き継ぎ、ノイズスケール sigma を soft_jaccard で追加学習。
    """

    def __init__(self, base_tt: TwoTowerModel, lambda_div: float):
        from src.model.models_006 import _DivAdapterModule
        from src.model.models_003 import DIV_LOSSES
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            lr=2e-3,
            epochs=30,
            batch_size=512,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=f"TT_soft_jaccard_l{str(lambda_div).replace('.','p')}",
        )
        self._base_tt = base_tt
        self.lambda_div = lambda_div
        self._div_fn = DIV_LOSSES["soft_jaccard"]
        self._adapter: _DivAdapterModule | None = None

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        from src.model.models_006 import _DivAdapterModule
        # ZCA と LogQ は base から引き継ぐ
        self._device = self._base_tt._device
        self.whitener = self._base_tt.whitener
        self.logq = self._base_tt.logq
        self.whitened_user_embs = self._base_tt.whitened_user_embs
        self.whitened_item_embs = self._base_tt.whitened_item_embs
        self.user_head = self._base_tt.user_head
        self.item_head = self._base_tt.item_head
        self._proj_item_embs = self._base_tt._proj_item_embs

        # soft_jaccard adapter を追加学習
        dev = self._device
        self._adapter = _DivAdapterModule(dim=self.hidden_dim).to(dev)
        self._train_adapter(train_pos)

    def _train_adapter(self, train_pos):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        # 全アイテムを item_head で投影（固定）
        X_proj_item = torch.from_numpy(self._proj_item_embs).float().to(dev)  # (N_items, hidden_dim)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items[:30]:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] Training diversity adapter: pairs={len(pairs_t)}, λ={self.lambda_div}")

        opt = torch.optim.Adam(self._adapter.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)

        self.user_head.eval()
        self.item_head.eval()

        for epoch in range(self.epochs):
            self._adapter.train()
            perm = torch.randperm(len(pairs_t))
            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                with torch.no_grad():
                    proj_u = self.user_head(X_user[u_idx])  # (B, hidden_dim)
                    proj_p = X_proj_item[p_idx]             # (B, hidden_dim)
                    proj_n = X_proj_item[n_idx]             # (B, hidden_dim)

                q1 = self._adapter(proj_u)
                q2 = self._adapter(proj_u)

                s1_pos = (q1 * proj_p).sum(-1)
                s1_neg = (q1 * proj_n).sum(-1)
                s2_pos = (q2 * proj_p).sum(-1)
                s2_neg = (q2 * proj_n).sum(-1)

                bpr = (-F.logsigmoid(s1_pos - s1_neg) - F.logsigmoid(s2_pos - s2_neg)).mean()
                # soft_jaccard は全アイテムのインデックスを使うので items を渡す
                d_loss = self._div_fn(q1, q2, X_proj_item)
                loss = bpr + self.lambda_div * d_loss

                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

        self._adapter.eval()
        log.info(f"[{self.name}] Adapter training complete.")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        dev = self._device
        u_t = torch.from_numpy(
            self.whitened_user_embs[user_idx: user_idx + 1]
        ).float().to(dev)
        with torch.no_grad():
            proj_u = self.user_head(u_t)
            q = self._adapter(proj_u)
        return q.cpu().numpy()[0]


def run_7c(best_tt: TwoTowerModel, train_pos, test_gt, user_embs, item_embs, device, n_total_items) -> dict:
    log.info("\n" + "="*60 + "\nSub-exp 7C: Best Two-Tower + soft_jaccard Adapter\n" + "="*60)
    results = {}
    for lam in LAMBDA_SWEEP:
        model = _TwoTowerWithSoftJaccard(best_tt, lambda_div=lam)
        results[model.name] = run_model_eval(
            model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
    return results


# ── Sub-exp 7D: Best Two-Tower + Post-training Noise ──────────────────────────

def run_7d(best_tt: TwoTowerModel, train_pos, test_gt, user_embs, item_embs, device, n_total_items) -> dict:
    log.info("\n" + "="*60 + "\nSub-exp 7D: Best Two-Tower + Post-Noise (inference-time)\n" + "="*60)
    results = {}
    for sigma in SIGMA_SWEEP:
        model = TwoTowerPostNoise(best_tt, sigma=sigma)
        # prepare() は base_tt の重みを引き継ぐだけなので高速
        model.prepare(train_pos, user_embs, item_embs, device=device)
        index = best_tt.build_index()
        eval_item_embs = best_tt._proj_item_embs

        seed_results = []
        for seed in SEEDS:
            m = evaluate_model_seed(
                model, test_gt, index, eval_item_embs,
                k=K, n_trials=N_TRIALS, seed=seed,
                n_total_items=n_total_items,
            )
            log.info(
                f"  seed={seed}  rc={m['recall_cum']:.4f}  ra={m['recall_avg']:.4f}  "
                f"hit={m['hit']:.4f}  ov={m['temporal_overlap']:.4f}"
            )
            seed_results.append(m)

        avg = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
        std = {k: float(np.std([r[k] for r in seed_results]))  for k in seed_results[0]}
        log.info(
            f"  [AVG] rc={avg['recall_cum']:.4f}±{std['recall_cum']:.4f}  "
            f"hit={avg['hit']:.4f}±{std['hit']:.4f}  "
            f"ov={avg['temporal_overlap']:.4f}±{std['temporal_overlap']:.4f}"
        )
        results[model.name] = {"mean": avg, "std": std, "per_seed": seed_results}
    return results


# ── Sub-exp 7E: Two-Tower + Training-time Noise ────────────────────────────────

def run_7e(best_depth: int, best_hidden_dim: int, train_pos, test_gt, user_embs, item_embs, device, n_total_items) -> dict:
    log.info("\n" + "="*60 + "\nSub-exp 7E: Two-Tower + Train-Noise (augmentation during training)\n" + "="*60)
    results = {}
    for sigma in SIGMA_SWEEP:
        model = TwoTowerTrainNoise(
            hidden_dim=best_hidden_dim,
            depth=best_depth,
            sigma=sigma,
            lr=1e-3,
            epochs=50,
            batch_size=1024,
            logit_scale=14.3,
            alpha=0.1,
        )
        results[model.name] = run_model_eval(
            model, test_gt, train_pos, user_embs, item_embs, device, n_total_items
        )
    return results


# ── Sub-exp 7D: Unified Plot ───────────────────────────────────────────────────

def plot_unified_007(all_results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#2d3142")
        ax.grid(True, color="#2d3142", linewidth=0.5, alpha=0.7)

    ax0, ax1 = axes

    COLOR_MAP = {
        "M0_strong":                  ("#ffffff",  "o", 12, "M0_strong (baseline)"),
        "TwoTower_d":                 ("#3498db",  "s",  7, "Two-Tower MLP"),
        "TT_soft_jaccard":            ("#9b59b6",  "^",  9, "Two-Tower + soft_jaccard"),
        "TwoTower_d1_h64_postnoise":  ("#e74c3c",  "v",  8, "Two-Tower + Post-Noise"),
        "TwoTower_d2_h64_postnoise":  ("#e74c3c",  "v",  8, "Two-Tower + Post-Noise"),
        "TwoTower_d3_h64_postnoise":  ("#e74c3c",  "v",  8, "Two-Tower + Post-Noise"),
        "TwoTower_d1_h128_postnoise": ("#e74c3c",  "v",  8, "Two-Tower + Post-Noise"),
        "TwoTower_d2_h128_postnoise": ("#e74c3c",  "v",  8, "Two-Tower + Post-Noise"),
        "TwoTower_d3_h128_postnoise": ("#e74c3c",  "v",  8, "Two-Tower + Post-Noise"),
        "postnoise":                  ("#e74c3c",  "v",  8, "Two-Tower + Post-Noise"),
        "trainnoise":                 ("#f39c12",  "D",  8, "Two-Tower + Train-Noise"),
    }

    def get_style(name):
        for key, (color, marker, size, label) in COLOR_MAP.items():  # type: ignore
            if name.startswith(key):
                return color, marker, size, label
        return "#aaaaaa", "o", 6, name

    plotted_labels = set()
    for name, r in all_results.items():
        m = r["mean"]
        s = r["std"]
        ov   = m["temporal_overlap"]
        div  = 1.0 - ov
        rc   = m["recall_cum"]
        ra   = m["recall_avg"]
        hit  = m["hit"]
        color, marker, size, label = get_style(name)

        lbl = label if label not in plotted_labels else "_nolegend_"
        plotted_labels.add(label)

        ax0.scatter(div, rc, c=color, marker=marker, s=size**2, alpha=0.85, label=lbl,
                    edgecolors="white", linewidths=0.3)
        ax0.annotate(name.split("_")[-1], (div, rc), fontsize=5.5, color="#cccccc",
                     xytext=(3, 3), textcoords="offset points")

        ax1.scatter(div, hit, c=color, marker=marker, s=size**2, alpha=0.85, label=lbl,
                    edgecolors="white", linewidths=0.3)

    for ax, ylabel, title in [
        (ax0, "recall_cum (N-trial union)",  "Diversity vs Recall_Cum — Plan 007"),
        (ax1, "hit@10 (avg per trial)",       "Diversity vs Hit@10 — Plan 007"),
    ]:
        ax.set_xlabel("Diversity (1 − temporal_overlap)", color="#e0e0e0", fontsize=11)
        ax.set_ylabel(ylabel, color="#e0e0e0", fontsize=11)
        ax.set_title(title, color="#ffffff", fontsize=13, pad=10)
        ax.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#444", labelcolor="#e0e0e0",
                  loc="lower right")

    plt.tight_layout()
    out_path = out_dir / "tradeoff_007_unified.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info(f"Saved unified tradeoff plot → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subexp", default="all",
                        choices=["all", "7A", "7B", "7C", "7D", "7E", "7F"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load data
    interactions, user_embs, item_embs, uid2idx, iid2idx = load_data(DATA_DIR)
    train_pos = get_train_pos(interactions, uid2idx, iid2idx)
    test_gt   = get_test_gt(interactions, uid2idx, iid2idx)
    n_total_items = len(item_embs)
    log.info(f"Train positives: {sum(len(v) for v in train_pos.values()):,} pairs")
    log.info(f"Test users: {len(test_gt):,}  Total items: {n_total_items:,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Load existing results if available (to allow partial reruns)
    results_path = OUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        log.info(f"Loaded {len(all_results)} existing results from {results_path}")

    run_all = args.subexp == "all"

    # 7A
    if run_all or args.subexp == "7A":
        r = run_7a(train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
        all_results.update(r)
        save_results(all_results, OUT_DIR)

    # 7B
    if run_all or args.subexp == "7B":
        r = run_7b(train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
        all_results.update(r)
        save_results(all_results, OUT_DIR)



    # Helper: find best Two-Tower (used by 7C/7D/7E)
    def _rebuild_best_tt():
        results_7b = {k: v for k, v in all_results.items()
                      if k.startswith("TwoTower_d") and "noise" not in k}
        if not results_7b:
            return None, None, None
        depth, hidden_dim = _find_best_tt(results_7b)
        best_tt = TwoTowerModel(
            hidden_dim=hidden_dim, depth=depth,
            lr=1e-3, epochs=50, batch_size=1024,
            logit_scale=14.3, alpha=0.1,
        )
        log.info(f"Re-training best Two-Tower (depth={depth}, hidden_dim={hidden_dim}) ...")
        best_tt.prepare(train_pos, user_embs, item_embs, device=args.device)
        return best_tt, depth, hidden_dim

    # 7C (requires 7B best model)
    if run_all or args.subexp == "7C":
        best_tt, depth, hidden_dim = _rebuild_best_tt()
        if best_tt is None:
            log.warning("No 7B results found. Run 7B first.")
        else:
            r = run_7c(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

    # 7D: Post-Noise
    if run_all or args.subexp == "7D":
        best_tt, depth, hidden_dim = _rebuild_best_tt()
        if best_tt is None:
            log.warning("No 7B results found. Run 7B first.")
        else:
            r = run_7d(best_tt, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

    # 7E: Train-Noise
    if run_all or args.subexp == "7E":
        results_7b = {k: v for k, v in all_results.items()
                      if k.startswith("TwoTower_d") and "noise" not in k}
        if not results_7b:
            log.warning("No 7B results found. Run 7B first.")
        else:
            depth, hidden_dim = _find_best_tt(results_7b)
            r = run_7e(depth, hidden_dim, train_pos, test_gt, user_embs, item_embs, args.device, n_total_items)
            all_results.update(r)
            save_results(all_results, OUT_DIR)

    # 7F: Unified Plot
    if run_all or args.subexp == "7F":
        plot_unified_007(all_results, OUT_DIR)

    log.info("\n✅ Plan 007 completed!")


if __name__ == "__main__":
    main()
