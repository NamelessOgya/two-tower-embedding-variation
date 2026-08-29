"""
Plan 015 Model: TwoTowerTrainedMultiHeadStratifiedPartition
------------------------------------------------------------
端対端学習型マルチヘッド層化分割モデル。
N_trials 個の層化抽出バケット I_1, ..., I_N にアラインするように、
N_trials 個のマルチヘッドプロジェクション W_1, ..., W_N を BPR 損失でファインチューニングする。
"""

from __future__ import annotations

import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from src.model.models_007 import TwoTowerModel

log = logging.getLogger(__name__)


class TwoTowerTrainedMultiHeadStratifiedPartition(TwoTowerModel):
    """
    ユーザーエンコーダーの出力層に N_trials 個の学習可能プロジェクションヘッド W_t を配置し、
    各バケット I_t に含まれる正解アイテムに対する BPR 損失で端対端学習（fine-tune）するモデル。
    推論時は試行 t において第 t ヘッド q_t = W_t(user_head(u)) で第 t バケット I_t 内から Top-K を推薦する（完全ステートレス）。
    """

    def __init__(
        self,
        base_tt: TwoTowerModel,
        n_trials: int = 10,
        n_clusters: int = 20,
        lr: float = 2e-3,
        epochs: int = 25,
        batch_size: int = 512,
    ):
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=f"TT_trained_multihead_stratified_partition_n{n_trials}",
        )
        self._base_tt = base_tt
        self.n_trials = n_trials
        self.n_clusters = n_clusters
        self.buckets: list[np.ndarray] = []
        self.item2bucket: dict[int, int] = {}

        dim = base_tt.hidden_dim
        self.heads = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_trials)])

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        self._device            = self._base_tt._device
        self.whitener           = self._base_tt.whitener
        self.logq               = self._base_tt.logq
        self.whitened_user_embs = self._base_tt.whitened_user_embs
        self.whitened_item_embs = self._base_tt.whitened_item_embs

        # user_head / item_head を deepcopy
        self.user_head = copy.deepcopy(self._base_tt.user_head)
        self.item_head = copy.deepcopy(self._base_tt.item_head)
        self._proj_item_embs = self._base_tt._proj_item_embs

        n_items = len(self.whitened_item_embs)

        # 1. K-Means 層化抽出バケット構築
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.whitened_item_embs)

        buckets_list = [[] for _ in range(self.n_trials)]
        rng = np.random.default_rng(42)
        for c in range(self.n_clusters):
            c_items = np.where(labels == c)[0]
            rng.shuffle(c_items)
            for i, item_idx in enumerate(c_items):
                buckets_list[i % self.n_trials].append(item_idx)

        self.buckets = [np.array(b, dtype=np.int64) for b in buckets_list]

        for b_idx, b_arr in enumerate(self.buckets):
            for i_idx in b_arr:
                self.item2bucket[int(i_idx)] = b_idx

        # 2. ヘッド初期化 (単位行列)
        for head in self.heads:
            nn.init.eye_(head.weight)

        log.info(
            f"[{self.name}] Fine-tuning {self.n_trials} multi-head projections on {n_items} items "
            f"partitioned into {self.n_trials} stratified buckets (epochs={self.epochs}, lr={self.lr})"
        )

        # 3. 端対端ファインチューニング
        self._train_multihead_buckets(train_pos)
        self._proj_item_embs = self._project_items()

    def _train_multihead_buckets(self, train_pos: dict[int, list[int]]):
        dev = self._device
        self.user_head.to(dev)
        self.heads.to(dev)
        self.item_head.to(dev)
        self.user_head.train()
        self.heads.train()
        self.item_head.eval()

        pairs = []
        for u_idx, pos_list in train_pos.items():
            for i_idx in pos_list:
                if i_idx in self.item2bucket:
                    b_idx = self.item2bucket[i_idx]
                    pairs.append((u_idx, i_idx, b_idx))

        pairs = np.array(pairs, dtype=np.int64)
        n_pairs = len(pairs)

        optimizer = torch.optim.Adam(
            list(self.user_head.parameters()) + list(self.heads.parameters()),
            lr=self.lr,
        )

        log.info(f"[{self.name}] Training pairs: {n_pairs}")

        u_embs_t = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        i_embs_t = torch.from_numpy(self.whitened_item_embs).float().to(dev)

        with torch.no_grad():
            proj_items_all = self.item_head(i_embs_t)

        for epoch in range(1, self.epochs + 1):
            perm = np.random.permutation(n_pairs)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_pairs, self.batch_size):
                end = min(start + self.batch_size, n_pairs)
                batch_idx = perm[start:end]
                batch_u = pairs[batch_idx, 0]
                batch_pos_i = pairs[batch_idx, 1]
                batch_b = pairs[batch_idx, 2]

                batch_neg_i = []
                for b_id in batch_b:
                    b_cand = self.buckets[b_id]
                    neg_cand = b_cand[np.random.randint(0, len(b_cand))]
                    batch_neg_i.append(neg_cand)
                batch_neg_i = np.array(batch_neg_i, dtype=np.int64)

                optimizer.zero_grad()

                u_batch = u_embs_t[batch_u]
                q_base = self.user_head(u_batch)

                q_multi = torch.zeros_like(q_base)
                for head_id in range(self.n_trials):
                    mask = (batch_b == head_id)
                    if np.any(mask):
                        mask_t = torch.from_numpy(mask).to(dev)
                        q_sub = self.heads[head_id](q_base[mask_t])
                        q_sub = F.normalize(q_sub, p=2, dim=1)
                        q_multi[mask_t] = q_sub

                pos_item_vecs = proj_items_all[batch_pos_i]
                neg_item_vecs = proj_items_all[batch_neg_i]

                pos_scores = (q_multi * pos_item_vecs).sum(dim=1)
                neg_scores = (q_multi * neg_item_vecs).sum(dim=1)

                if self.logq is not None:
                    pos_pen = torch.from_numpy(self.logq.get_penalties(batch_pos_i)).float().to(dev)
                    neg_pen = torch.from_numpy(self.logq.get_penalties(batch_neg_i)).float().to(dev)
                    pos_scores = self.logit_scale * pos_scores - self.alpha * pos_pen
                    neg_scores = self.logit_scale * neg_scores - self.alpha * neg_pen

                loss = -F.logsigmoid(pos_scores - neg_scores).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 5 == 0 or epoch == self.epochs:
                avg_loss = epoch_loss / max(1, n_batches)
                log.info(f"[{self.name}] epoch={epoch}/{self.epochs}  loss={avg_loss:.4f}")

        self.user_head.eval()
        self.heads.eval()

    def recommend(
        self,
        user_idx: int,
        trial: int,
        rng: np.random.Generator,
        index,
        k: int,
        candidate_pool_size: int = 200,
    ) -> list[int]:
        head_idx = trial % self.n_trials
        cand_items = self.buckets[head_idx]

        dev = self._device
        u_t = torch.from_numpy(self.whitened_user_embs[user_idx: user_idx + 1]).float().to(dev)
        with torch.no_grad():
            q_base = self.user_head(u_t)
            q_head = self.heads[head_idx](q_base)
            q_head = F.normalize(q_head, p=2, dim=1)
            q = q_head.cpu().numpy()[0]

        cand_embs = self._proj_item_embs[cand_items]
        cand_scores = (cand_embs @ q).astype(np.float64)

        if self.logq is not None:
            penalties = self.logq.get_penalties(cand_items)
            cand_scores = self.logit_scale * cand_scores - self.alpha * penalties

        topk_local = np.argpartition(cand_scores, -min(k, len(cand_scores)))[-min(k, len(cand_scores)):]
        topk_sorted = topk_local[np.argsort(-cand_scores[topk_local])]
        return cand_items[topk_sorted].tolist()
