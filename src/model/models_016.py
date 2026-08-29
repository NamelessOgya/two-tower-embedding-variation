"""
Plan 016 Model: TwoTowerTwoStagePlackettLuce
--------------------------------------------
2段階推薦アーキテクチャにおける Plackett-Luce 確率的サンプリングモデル。
Stage 1: FAISS 近傍検索によりスコア上位 M 件 (例: M=200) の候補プール C_u を取得
Stage 2: 候補プール C_u 内で Gumbel-Top-K Trick による Plackett-Luce 確率的非復元サンプリングで Top-K スレートを生成
"""

from __future__ import annotations

import logging
import numpy as np
import torch

from src.model.models_007 import TwoTowerModel

log = logging.getLogger(__name__)


class TwoTowerTwoStagePlackettLuce(TwoTowerModel):
    """
    Two-Stage Recommendation Architecture with Plackett-Luce Sampling.
    Stage 1: Retrieve top-M candidate pool C_u using Two-Tower FAISS index.
    Stage 2: Sample top-K items from C_u using Gumbel-Top-K trick under temperature tau.
    """

    def __init__(
        self,
        base_tt: TwoTowerModel,
        tau: float = 1.0,
        m_candidates: int = 200,
        name_suffix: str = "",
    ):
        tau_str = str(tau).replace(".", "p")
        name = f"TT_2stage_PL_M{m_candidates}_tau{tau_str}{name_suffix}"
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=name,
        )
        self._base_tt = base_tt
        self.tau = tau
        self.m_candidates = m_candidates

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        self._device            = self._base_tt._device
        self.whitener           = self._base_tt.whitener
        self.logq               = self._base_tt.logq
        self.whitened_user_embs = self._base_tt.whitened_user_embs
        self.whitened_item_embs = self._base_tt.whitened_item_embs
        self.user_head          = self._base_tt.user_head
        self.item_head          = self._base_tt.item_head
        self._proj_item_embs   = self._base_tt._proj_item_embs

        log.info(
            f"[{self.name}] Initialized 2-Stage Plackett-Luce (M={self.m_candidates}, tau={self.tau})"
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
        q = self._base_tt.get_query_vector(user_idx, trial, rng)

        # Stage 1: FAISS から上位 M 件の候補プール C_u を取得
        M = max(self.m_candidates, k)
        q_search = q.reshape(1, -1).astype(np.float32)
        _, cand_indices = index.search(q_search, M)
        cand_items = cand_indices[0]  # (M,)

        # Stage 2: 候補プール C_u 内でスコア計算
        cand_embs = self._proj_item_embs[cand_items]
        cand_scores = (cand_embs @ q).astype(np.float64)

        if self.logq is not None:
            penalties = self.logq.get_penalties(cand_items)
            cand_scores = self.logit_scale * cand_scores - self.alpha * penalties

        # Plackett-Luce サンプリング (Gumbel-Top-K Trick)
        if self.tau <= 1e-6:
            # tau -> 0: 決定論的 Top-K
            topk_local = np.argpartition(cand_scores, -min(k, len(cand_scores)))[-min(k, len(cand_scores)):]
            topk_sorted = topk_local[np.argsort(-cand_scores[topk_local])]
        else:
            # Gumbel(0, 1) ノイズ生成: -log(-log(U)) where U ~ Uniform(0, 1)
            u_rand = rng.uniform(low=1e-10, high=1.0 - 1e-10, size=len(cand_scores))
            gumbel_noise = -np.log(-np.log(u_rand))

            # Gumbel-Top-K スコア: (score / tau) + Gumbel
            sampled_scores = (cand_scores / self.tau) + gumbel_noise

            topk_local = np.argpartition(sampled_scores, -min(k, len(sampled_scores)))[-min(k, len(sampled_scores)):]
            topk_sorted = topk_local[np.argsort(-sampled_scores[topk_local])]

        return cand_items[topk_sorted].tolist()
