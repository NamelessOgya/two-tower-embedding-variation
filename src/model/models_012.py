"""
Plan 012 Models: Advanced Soft-Jaccard Variants & Item Partition Baseline
--------------------------------------------------------------------------
Plan 008〜011 で最も優れた探索性能を示した soft_jaccard DivLoss の拡張・改良モデル群、
および全アイテムを試行ごとに分割する Item Partition ベースライン。

12A: TwoTowerTopKSoftJaccard
    - 上位 K_loss (例: 30件) の確率分布のみを用いて Soft Jaccard 損失を計算。

12B: TwoTowerAdaptiveSoftJaccard
    - 次元別のノイズスケール log_sigma を定義し、BPR + Soft Jaccard で end-to-end 学習。

12C: TwoTowerSemanticSoftJaccard
    - アイテム埋め込み類似度行列 S_ij を考慮した Soft Jaccard 損失。

12E (Baseline): TwoTowerItemPartition
    - 全アイテムを N_trials 個の重複のないバケットにランダム分割。
    - 試行 t では第 t バケット内のアイテムのみから Top-K を推薦（Overlap = 0.0, Diversity = 1.0）。
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Callable

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.models_007 import TwoTowerModel, MLPHead
from src.model.models_003 import DIV_LOSSES, NEEDS_ITEMS

log = logging.getLogger(__name__)


# ── 12E: Item Partition Baseline (全アイテムを試行ごとに分割) ─────────────────

class TwoTowerItemPartition(TwoTowerModel):
    """
    全アイテム集合を N_trials 個の重複しないバケットに分割し、
    試行 t では第 t バケットに含まれるアイテム群のみから Top-K を推薦するモデル。
    試行間のアイテム重複が完全にゼロ (Overlap = 0.0, Diversity = 1.0) になる決定論的・強制分割ベースライン。
    """

    def __init__(
        self,
        base_tt: TwoTowerModel,
        n_trials: int = 10,
    ):
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=f"TT_item_partition_n{n_trials}",
        )
        self._base_tt = base_tt
        self.n_trials = n_trials
        self.buckets: list[np.ndarray] = []  # 各試行に割り当てるアイテム index 配列

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        self._device            = self._base_tt._device
        self.whitener           = self._base_tt.whitener
        self.logq               = self._base_tt.logq
        self.whitened_user_embs = self._base_tt.whitened_user_embs
        self.whitened_item_embs = self._base_tt.whitened_item_embs
        self.user_head          = self._base_tt.user_head
        self.item_head          = self._base_tt.item_head
        self._proj_item_embs   = self._base_tt._proj_item_embs

        n_items = len(self.whitened_item_embs)
        # 固定シードでアイテムをシャッフルし n_trials 個のバケットに分割
        perm_rng = np.random.default_rng(42)
        shuffled_items = perm_rng.permutation(n_items)
        self.buckets = np.array_split(shuffled_items, self.n_trials)
        log.info(f"[{self.name}] Partitioned {n_items} items into {self.n_trials} buckets (sizes: {[len(b) for b in self.buckets]})")

    def recommend(
        self,
        user_idx: int,
        trial: int,
        rng: np.random.Generator,
        index: faiss.IndexFlatIP,
        k: int,
        candidate_pool_size: int = 200,
    ) -> list[int]:
        bucket_idx = trial % self.n_trials
        cand_items = self.buckets[bucket_idx]

        # クエリベクトル取得
        dev = self._device
        u_t = torch.from_numpy(self.whitened_user_embs[user_idx: user_idx + 1]).float().to(dev)
        with torch.no_grad():
            q = self.user_head(u_t).cpu().numpy()[0]

        # 該当バケット内のアイテムの投影ベクトルとの内積スコアを計算
        cand_embs = self._proj_item_embs[cand_items]  # (N_bucket, hidden_dim)
        cand_scores = (cand_embs @ q).astype(np.float64)

        # LogQ 補正
        if self.logq is not None:
            penalties = self.logq.get_penalties(cand_items)
            cand_scores = self.logit_scale * cand_scores - self.alpha * penalties

        # スコア Top-K を選択
        topk_local = np.argpartition(cand_scores, -min(k, len(cand_scores)))[-min(k, len(cand_scores)):]
        topk_sorted = topk_local[np.argsort(-cand_scores[topk_local])]
        return list(cand_items[topk_sorted])


# ── 12A: Top-K Truncated Soft Jaccard ─────────────────────────────────────────

def div_topk_soft_jaccard(
    q1: torch.Tensor,
    q2: torch.Tensor,
    items: torch.Tensor,
    T: float = 0.1,
    topk: int = 30,
) -> torch.Tensor:
    s1 = (items @ q1.T) / T
    s2 = (items @ q2.T) / T

    topk_vals1, _ = torch.topk(s1, topk, dim=0)
    thresh1 = topk_vals1[-1:, :]
    mask1 = s1 < thresh1
    s1_masked = s1.masked_fill(mask1, -1e9)

    topk_vals2, _ = torch.topk(s2, topk, dim=0)
    thresh2 = topk_vals2[-1:, :]
    mask2 = s2 < thresh2
    s2_masked = s2.masked_fill(mask2, -1e9)

    p1 = F.softmax(s1_masked, dim=0)
    p2 = F.softmax(s2_masked, dim=0)

    inter = torch.min(p1, p2).sum(0).mean()
    union = torch.max(p1, p2).sum(0).mean()
    return inter / (union + 1e-10)


class TwoTowerTopKSoftJaccard(TwoTowerModel):
    """Sub-exp 12A: Top-K 集中型 Soft Jaccard モデル"""

    def __init__(
        self,
        base_tt: TwoTowerModel,
        topk: int = 30,
        lambda_div: float = 0.1,
        sigma: float = 0.05,
        lr: float = 2e-3,
        epochs: int = 30,
        batch_size: int = 512,
    ):
        lam_str = str(lambda_div).replace(".", "p")
        sig_str = str(sigma).replace(".", "p")
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=f"TT_topk{topk}_sj_l{lam_str}_s{sig_str}",
        )
        self._base_tt = base_tt
        self.topk = topk
        self.lambda_div = lambda_div
        self.sigma = sigma

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        self._device            = self._base_tt._device
        self.whitener           = self._base_tt.whitener
        self.logq               = self._base_tt.logq
        self.whitened_user_embs = self._base_tt.whitened_user_embs
        self.whitened_item_embs = self._base_tt.whitened_item_embs
        self._proj_item_embs    = self._base_tt._proj_item_embs

        self.user_head = copy.deepcopy(self._base_tt.user_head)
        self.item_head = copy.deepcopy(self._base_tt.item_head)

        log.info(f"[{self.name}] Training TopK-SoftJaccard: topk={self.topk}, "
                 f"lambda={self.lambda_div}, sigma={self.sigma}")
        self._train_topk_sj(train_pos)
        self._proj_item_embs = self._project_items()

    def _train_topk_sj(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)

        opt = torch.optim.Adam(self.user_head.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)

        self.user_head.train()
        self.item_head.eval()

        n_div_items = 500
        div_item_idx = torch.randperm(N_items)[:n_div_items]
        X_div = X_item[div_item_idx]
        with torch.no_grad():
            proj_div = self.item_head(X_div)

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                u_emb = X_user[u_idx]
                eps1 = torch.randn_like(u_emb) * self.sigma
                eps2 = torch.randn_like(u_emb) * self.sigma

                q1 = self.user_head(F.normalize(u_emb + eps1, p=2, dim=-1))
                q2 = self.user_head(F.normalize(u_emb + eps2, p=2, dim=-1))

                with torch.no_grad():
                    proj_p = self.item_head(X_item[p_idx])
                    proj_n = self.item_head(X_item[n_idx])

                s_pos1 = (q1 * proj_p).sum(-1)
                s_neg1 = (q1 * proj_n).sum(-1)
                s_pos2 = (q2 * proj_p).sum(-1)
                s_neg2 = (q2 * proj_n).sum(-1)

                bpr_loss = 0.5 * (-F.logsigmoid(s_pos1 - s_neg1).mean() + -F.logsigmoid(s_pos2 - s_neg2).mean())
                div_loss = div_topk_soft_jaccard(q1, q2, proj_div, T=0.1, topk=self.topk)

                loss = bpr_loss + self.lambda_div * div_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            sched.step()
        self.user_head.eval()

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        dev = self._device
        u_t = torch.from_numpy(self.whitened_user_embs[user_idx: user_idx + 1]).float().to(dev)
        if self.sigma > 0:
            noise = torch.from_numpy(rng.normal(0, self.sigma, size=u_t.shape)).float().to(dev)
            u_t = F.normalize(u_t + noise, p=2, dim=-1)
        with torch.no_grad():
            q = self.user_head(u_t).cpu().numpy()[0]
        return q.astype(np.float32)


# ── 12B: Learned Per-Dimension Noise (Adaptive Soft-Jaccard) ───────────────────

class TwoTowerAdaptiveSoftJaccard(TwoTowerModel):
    """
    Sub-exp 12B: Learned Per-Dimension Noise (Adaptive Soft Jaccard)
    次元別のノイズスケール log_sigma を定義し、Soft Jaccard 損失 + BPR で学習。
    """

    def __init__(
        self,
        base_tt: TwoTowerModel,
        lambda_div: float = 0.1,
        init_sigma: float = 0.05,
        lr: float = 2e-3,
        epochs: int = 30,
        batch_size: int = 512,
    ):
        lam_str = str(lambda_div).replace(".", "p")
        init_str = str(init_sigma).replace(".", "p")
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=f"TT_adaptive_sj_l{lam_str}_init{init_str}",
        )
        self._base_tt = base_tt
        self.lambda_div = lambda_div
        self.init_sigma = init_sigma
        self.log_sigma_param: Optional[nn.Parameter] = None

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        self._device            = self._base_tt._device
        self.whitener           = self._base_tt.whitener
        self.logq               = self._base_tt.logq
        self.whitened_user_embs = self._base_tt.whitened_user_embs
        self.whitened_item_embs = self._base_tt.whitened_item_embs
        self._proj_item_embs    = self._base_tt._proj_item_embs

        self.user_head = copy.deepcopy(self._base_tt.user_head)
        self.item_head = copy.deepcopy(self._base_tt.item_head)

        dim = self.whitened_user_embs.shape[1]
        init_val = float(np.log(np.exp(self.init_sigma) - 1.0 + 1e-6))
        self.log_sigma_param = nn.Parameter(torch.full((dim,), init_val, device=self._device))

        log.info(f"[{self.name}] Training Adaptive SoftJaccard: lambda={self.lambda_div}, init_sigma={self.init_sigma}")
        self._train_adaptive_sj(train_pos)
        self._proj_item_embs = self._project_items()

    def _train_adaptive_sj(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)

        opt = torch.optim.Adam(list(self.user_head.parameters()) + [self.log_sigma_param], lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)

        self.user_head.train()
        self.item_head.eval()

        n_div_items = 500
        div_item_idx = torch.randperm(N_items)[:n_div_items]
        X_div = X_item[div_item_idx]
        with torch.no_grad():
            proj_div = self.item_head(X_div)

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss = 0.0
            n_batches = 0
            sigmas = F.softplus(self.log_sigma_param)

            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                u_emb = X_user[u_idx]
                eps1 = torch.randn_like(u_emb) * sigmas
                eps2 = torch.randn_like(u_emb) * sigmas

                q1 = self.user_head(F.normalize(u_emb + eps1, p=2, dim=-1))
                q2 = self.user_head(F.normalize(u_emb + eps2, p=2, dim=-1))

                with torch.no_grad():
                    proj_p = self.item_head(X_item[p_idx])
                    proj_n = self.item_head(X_item[n_idx])

                s_pos1 = (q1 * proj_p).sum(-1)
                s_neg1 = (q1 * proj_n).sum(-1)
                s_pos2 = (q2 * proj_p).sum(-1)
                s_neg2 = (q2 * proj_n).sum(-1)

                bpr_loss = 0.5 * (-F.logsigmoid(s_pos1 - s_neg1).mean() + -F.logsigmoid(s_pos2 - s_neg2).mean())

                s1 = (proj_div @ q1.T) / 0.1
                s2 = (proj_div @ q2.T) / 0.1
                p1 = F.softmax(s1, dim=0)
                p2 = F.softmax(s2, dim=0)
                inter = torch.min(p1, p2).sum(0).mean()
                union = torch.max(p1, p2).sum(0).mean()
                div_loss = inter / (union + 1e-10)

                loss = bpr_loss + self.lambda_div * div_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            sched.step()
            sig_mean = F.softplus(self.log_sigma_param).mean().item()
            if (epoch + 1) % 10 == 0:
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs} loss={epoch_loss/n_batches:.4f} mean_sigma={sig_mean:.4f}")

        self.user_head.eval()

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        dev = self._device
        u_t = torch.from_numpy(self.whitened_user_embs[user_idx: user_idx + 1]).float().to(dev)
        with torch.no_grad():
            sigmas = F.softplus(self.log_sigma_param)
            noise_np = rng.normal(0.0, 1.0, size=u_t.shape).astype(np.float32)
            noise = torch.from_numpy(noise_np).to(dev) * sigmas
            u_noisy = F.normalize(u_t + noise, p=2, dim=-1)
            q = self.user_head(u_noisy).cpu().numpy()[0]
        return q.astype(np.float32)


# ── 12C: Semantic-aware Soft Jaccard ─────────────────────────────────────────

def div_semantic_soft_jaccard(
    q1: torch.Tensor,
    q2: torch.Tensor,
    items: torch.Tensor,
    T: float = 0.1,
) -> torch.Tensor:
    s1 = (items @ q1.T) / T
    s2 = (items @ q2.T) / T
    p1 = F.softmax(s1, dim=0)
    p2 = F.softmax(s2, dim=0)

    S = (items @ items.T).clamp(min=0.0)

    Sp1 = S @ p1
    Sp2 = S @ p2

    p1_S_p2 = (p1 * Sp2).sum(0)
    p1_S_p1 = (p1 * Sp1).sum(0)
    p2_S_p2 = (p2 * Sp2).sum(0)

    inter = p1_S_p2
    union = p1_S_p1 + p2_S_p2 - p1_S_p2
    return (inter / (union + 1e-10)).mean()


class TwoTowerSemanticSoftJaccard(TwoTowerModel):
    """Sub-exp 12C: 意味的類似度考慮型 Soft Jaccard モデル"""

    def __init__(
        self,
        base_tt: TwoTowerModel,
        lambda_div: float = 0.1,
        sigma: float = 0.05,
        lr: float = 2e-3,
        epochs: int = 30,
        batch_size: int = 512,
    ):
        lam_str = str(lambda_div).replace(".", "p")
        sig_str = str(sigma).replace(".", "p")
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=f"TT_semantic_sj_l{lam_str}_s{sig_str}",
        )
        self._base_tt = base_tt
        self.lambda_div = lambda_div
        self.sigma = sigma

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        self._device            = self._base_tt._device
        self.whitener           = self._base_tt.whitener
        self.logq               = self._base_tt.logq
        self.whitened_user_embs = self._base_tt.whitened_user_embs
        self.whitened_item_embs = self._base_tt.whitened_item_embs
        self._proj_item_embs    = self._base_tt._proj_item_embs

        self.user_head = copy.deepcopy(self._base_tt.user_head)
        self.item_head = copy.deepcopy(self._base_tt.item_head)

        log.info(f"[{self.name}] Training Semantic SoftJaccard: lambda={self.lambda_div}, sigma={self.sigma}")
        self._train_semantic_sj(train_pos)
        self._proj_item_embs = self._project_items()

    def _train_semantic_sj(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)

        opt = torch.optim.Adam(self.user_head.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)

        self.user_head.train()
        self.item_head.eval()

        n_div_items = 400
        div_item_idx = torch.randperm(N_items)[:n_div_items]
        X_div = X_item[div_item_idx]
        with torch.no_grad():
            proj_div = self.item_head(X_div)

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                u_emb = X_user[u_idx]
                eps1 = torch.randn_like(u_emb) * self.sigma
                eps2 = torch.randn_like(u_emb) * self.sigma

                q1 = self.user_head(F.normalize(u_emb + eps1, p=2, dim=-1))
                q2 = self.user_head(F.normalize(u_emb + eps2, p=2, dim=-1))

                with torch.no_grad():
                    proj_p = self.item_head(X_item[p_idx])
                    proj_n = self.item_head(X_item[n_idx])

                s_pos1 = (q1 * proj_p).sum(-1)
                s_neg1 = (q1 * proj_n).sum(-1)
                s_pos2 = (q2 * proj_p).sum(-1)
                s_neg2 = (q2 * proj_n).sum(-1)

                bpr_loss = 0.5 * (-F.logsigmoid(s_pos1 - s_neg1).mean() + -F.logsigmoid(s_pos2 - s_neg2).mean())
                div_loss = div_semantic_soft_jaccard(q1, q2, proj_div, T=0.1)

                loss = bpr_loss + self.lambda_div * div_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            sched.step()
        self.user_head.eval()

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        dev = self._device
        u_t = torch.from_numpy(self.whitened_user_embs[user_idx: user_idx + 1]).float().to(dev)
        if self.sigma > 0:
            noise = torch.from_numpy(rng.normal(0, self.sigma, size=u_t.shape)).float().to(dev)
            u_t = F.normalize(u_t + noise, p=2, dim=-1)
        with torch.no_grad():
            q = self.user_head(u_t).cpu().numpy()[0]
        return q.astype(np.float32)
