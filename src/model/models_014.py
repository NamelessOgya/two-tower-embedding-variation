"""
Plan 014 Models: Stateless High-Precision & High-Diversity Recommendation Models
---------------------------------------------------------------------------------
1. TwoTowerSemanticStratifiedPartition
   - K-Means 層化抽出により全アイテムを N_trials 個のバケットに分配。
   - 過去履歴非依存（完全ステートレス）かつ試行間重複ゼロ (Overlap = 0.0)。

2. TwoTowerMultiHeadStratifiedPartition
   - 意味論的層化分割に、ユーザーエンコーダー側のマルチヘッド直交射影 W_t を組み合わせる。
   - 第 t 試行では第 t ヘッドベクトル q_t で第 t バケットを Top-K 検索。
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from sklearn.cluster import KMeans

from src.model.models_007 import TwoTowerModel

log = logging.getLogger(__name__)


class TwoTowerSemanticStratifiedPartition(TwoTowerModel):
    """
    アイテム埋め込み空間（whitened_item_embs）上で K-Means クラスタリングを行い、
    各クラスタからアイテムを均等（Stratified）に N_trials 個のバケットへ分配するモデル。
    各バケットに全カテゴリの代表品が均等に含まれるため、毎回の枠精度（Total Slate Precision）が向上する。
    """

    def __init__(
        self,
        base_tt: TwoTowerModel,
        n_trials: int = 10,
        n_clusters: int = 20,
    ):
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=f"TT_semantic_stratified_partition_n{n_trials}",
        )
        self._base_tt = base_tt
        self.n_trials = n_trials
        self.n_clusters = n_clusters
        self.buckets: list[np.ndarray] = []

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

        # 1. K-Means クラスタリング
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.whitened_item_embs)

        # 2. クラスタごとの層化抽出（Stratified Round-Robin Split）
        buckets_list = [[] for _ in range(self.n_trials)]
        rng = np.random.default_rng(42)
        for c in range(self.n_clusters):
            c_items = np.where(labels == c)[0]
            rng.shuffle(c_items)
            for i, item_idx in enumerate(c_items):
                buckets_list[i % self.n_trials].append(item_idx)

        self.buckets = [np.array(b, dtype=np.int64) for b in buckets_list]
        log.info(
            f"[{self.name}] Stratified partitioned {n_items} items into {self.n_trials} buckets "
            f"(sizes: {[len(b) for b in self.buckets]})"
        )

    def recommend(
        self,
        user_idx: int,
        trial: int,
        rng: np.random.Generator,
        index,
        k: int,
        candidate_pool_size: int = 200,
    ) -> list[int]:
        bucket_idx = trial % self.n_trials
        cand_items = self.buckets[bucket_idx]

        dev = self._device
        u_t = torch.from_numpy(self.whitened_user_embs[user_idx: user_idx + 1]).float().to(dev)
        with torch.no_grad():
            q = self.user_head(u_t).cpu().numpy()[0]

        cand_embs = self._proj_item_embs[cand_items]
        cand_scores = (cand_embs @ q).astype(np.float64)

        if self.logq is not None:
            penalties = self.logq.get_penalties(cand_items)
            cand_scores = self.logit_scale * cand_scores - self.alpha * penalties

        topk_local = np.argpartition(cand_scores, -min(k, len(cand_scores)))[-min(k, len(cand_scores)):]
        topk_sorted = topk_local[np.argsort(-cand_scores[topk_local])]
        return cand_items[topk_sorted].tolist()


class TwoTowerMultiHeadStratifiedPartition(TwoTowerModel):
    """
    意味論的層化バケット分割と、ユーザー側の直交マルチヘッド射影 W_t を組み合わせたモデル。
    試行 t では第 t ヘッドで得られるユーザーベクトル q_t で第 t バケットを Top-K 検索する。
    """

    def __init__(
        self,
        base_tt: TwoTowerModel,
        n_trials: int = 10,
        n_clusters: int = 20,
    ):
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=f"TT_multihead_stratified_partition_n{n_trials}",
        )
        self._base_tt = base_tt
        self.n_trials = n_trials
        self.n_clusters = n_clusters
        self.buckets: list[np.ndarray] = []
        self.head_matrices: list[np.ndarray] = []

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

        # 1. K-Means 層化抽出バケットの作成
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

        # 2. 直交マルチヘッド変換行列の生成 (QR 分解による等距離回転)
        dim = self.hidden_dim
        head_mats = []
        qr_rng = np.random.default_rng(42)
        for _ in range(self.n_trials):
            A = qr_rng.standard_normal((dim, dim))
            Q, R = np.linalg.qr(A)
            Q = Q * np.sign(np.diag(R))
            head_mats.append(Q.astype(np.float32))
        self.head_matrices = head_mats

        log.info(
            f"[{self.name}] Multi-head Stratified partitioned {n_items} items into {self.n_trials} buckets "
            f"with {self.n_trials} orthogonal heads."
        )

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
            q_base = self.user_head(u_t).cpu().numpy()[0]

        # ヘッド変換 (回転)
        q = q_base @ self.head_matrices[head_idx]
        norm_q = np.linalg.norm(q)
        if norm_q > 0:
            q = q / norm_q

        cand_embs = self._proj_item_embs[cand_items]
        cand_scores = (cand_embs @ q).astype(np.float64)

        if self.logq is not None:
            penalties = self.logq.get_penalties(cand_items)
            cand_scores = self.logit_scale * cand_scores - self.alpha * penalties

        topk_local = np.argpartition(cand_scores, -min(k, len(cand_scores)))[-min(k, len(cand_scores)):]
        topk_sorted = topk_local[np.argsort(-cand_scores[topk_local])]
        return cand_items[topk_sorted].tolist()
