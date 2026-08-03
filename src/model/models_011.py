"""
Plan 011 Models
---------------
Sub-exp 11A: TwoTowerDPP  — 推論時に Determinantal Point Process (DPP) で K アイテムを選択
Sub-exp 11B: TwoTowerMultiHead — 複数 user ヘッドを訓練時多様性損失で分離させる手法

どちらも:
  - 入力インターフェース変更なし（mE5 + ZCA whitened user embedding）
  - 推論時に過去試行の記憶不要
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.models_007 import TwoTowerModel, MLPHead
from src.model.models_003 import div_soft_jaccard

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-exp 11A: TwoTowerDPP
# ═══════════════════════════════════════════════════════════════════════════════

class TwoTowerDPP(TwoTowerModel):
    """
    既存の Two-Tower（または soft_jaccard fine-tuned）モデルをベースに、
    推論時の K アイテム選択を DPP (Determinantal Point Process) MAP 推定で行う。

    通常の top-K 選択:
        candidates → top-K by relevance score

    DPP MAP 選択（本クラス）:
        candidates → greedy DPP MAP → K items that balance relevance × diversity

    DPP カーネル:
        L_ij = quality_i * kernel(item_i, item_j) * quality_j
        kernel = cosine similarity between projected item embeddings

    再学習不要。recommend() のみ変更。
    """

    def __init__(
        self,
        base_tt: TwoTowerModel,
        sigma: float = 0.0,
        candidate_size: int = 200,
        name: Optional[str] = None,
    ):
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
        )
        sig_str = str(sigma).replace(".", "p")
        self.name = name or f"{base_tt.name}_dpp_s{sig_str}"
        self._base_tt     = base_tt
        self.sigma        = sigma
        self.candidate_size = candidate_size

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        # base_tt の全情報を引き継ぐ（再学習なし）
        self._device             = self._base_tt._device
        self.whitener            = self._base_tt.whitener
        self.logq                = self._base_tt.logq
        self.whitened_user_embs  = self._base_tt.whitened_user_embs
        self.whitened_item_embs  = self._base_tt.whitened_item_embs
        self.user_head           = self._base_tt.user_head
        self.item_head           = self._base_tt.item_head
        self._proj_item_embs     = self._base_tt._proj_item_embs
        log.info(f"[{self.name}] DPP wrapper: candidate_size={self.candidate_size}, sigma={self.sigma}")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        dev = self._device
        u_t = torch.from_numpy(
            self.whitened_user_embs[user_idx: user_idx + 1]
        ).float().to(dev)
        with torch.no_grad():
            proj = self.user_head(u_t)
        q = proj.cpu().numpy()[0]
        if self.sigma > 0:
            noise = rng.normal(0, self.sigma, size=q.shape).astype(np.float32)
            q = q + noise
            norm = np.linalg.norm(q)
            if norm > 1e-9:
                q = q / norm
        return q.astype(np.float32)

    def recommend(
        self,
        user_idx: int,
        trial: int,
        rng: np.random.Generator,
        index: faiss.IndexFlatIP,
        k: int,
        candidate_pool_size: int = 200,
    ) -> list[int]:
        q = self.get_query_vector(user_idx, trial, rng).reshape(1, -1)
        n_cands = min(self.candidate_size, index.ntotal)
        scores, I = index.search(q, n_cands)
        cand_scores  = scores[0].astype(np.float64)
        cand_indices = I[0]

        # LogQ 補正
        if self.logq is not None:
            penalties    = self.logq.get_penalties(cand_indices)
            cand_scores  = self.logit_scale * cand_scores - self.alpha * penalties

        # 候補アイテムの投影ベクトル（L2 正規化済み）
        cand_embs = self._proj_item_embs[cand_indices]  # (n_cands, hidden_dim)

        # DPP MAP 選択
        selected_local = self._dpp_greedy(cand_scores, cand_embs, k)
        return list(cand_indices[selected_local])

    @staticmethod
    def _dpp_greedy(
        quality_scores: np.ndarray,
        item_embs: np.ndarray,
        k: int,
    ) -> list[int]:
        """
        Greedy MAP inference for Quality-Diversity DPP.

        L_ij = q_i * S_ij * q_j
            q_i = normalized quality score of item i
            S_ij = cosine similarity between item i and j (shifted to [0,1])

        Marginal gain of adding item i given already selected S:
            gain_i = L_ii - L_{i,S} @ inv(L_{S,S}) @ L_{S,i}

        Complexity: O(K^2 * N) — negligible for K=10, N=200.
        """
        N = len(quality_scores)

        # Quality を [0,1] に正規化
        q_min = quality_scores.min()
        q_max = quality_scores.max()
        q = (quality_scores - q_min) / (q_max - q_min + 1e-9)

        # Cosine similarity kernel (item_embs は L2 正規化済み)
        S = (item_embs @ item_embs.T).astype(np.float64)
        S = (S + 1.0) / 2.0  # [-1,1] → [0,1]

        # DPP カーネル行列 L_ij = q_i * S_ij * q_j
        L = np.outer(q, q) * S
        L += 1e-8 * np.eye(N)  # 数値安定性

        selected  = []
        remaining = list(range(N))

        for _ in range(k):
            if not remaining:
                break

            if len(selected) == 0:
                # 初回: 対角最大（= 最高 quality）
                gains = np.array([L[i, i] for i in remaining])
            else:
                # Schur complement で限界ゲインを計算
                L_SS = L[np.ix_(selected, selected)]
                try:
                    L_SS_inv = np.linalg.inv(L_SS)
                except np.linalg.LinAlgError:
                    L_SS_inv = np.linalg.pinv(L_SS)
                gains = np.array([
                    max(L[i, i] - L[i, selected] @ L_SS_inv @ L[selected, i], 1e-12)
                    for i in remaining
                ])

            best = remaining[int(np.argmax(gains))]
            selected.append(best)
            remaining.remove(best)

        return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-exp 11B: TwoTowerMultiHead
# ═══════════════════════════════════════════════════════════════════════════════

class TwoTowerMultiHead(TwoTowerModel):
    """
    複数の user ヘッドを持つ Two-Tower モデル。

    訓練:
        各ヘッドに対して BPR 損失を計算 + ヘッド間 soft_jaccard 多様性損失で分離を促進。
        L = (1/M) Σ_m BPR(q_m) + λ * (1/C(M,2)) Σ_{i<j} SoftJaccard(q_i, q_j)

    推論:
        trial ごとにランダムに1つのヘッドを選択してクエリを生成。
        → 記憶不要・インターフェース変更なし。
        オプションで小さなノイズ (sigma) を追加可能。

    入力: 同じ user embedding（mE5 + ZCA）— インターフェース変更なし。
    """

    def __init__(
        self,
        base_tt: TwoTowerModel,
        n_heads: int = 3,
        lambda_div: float = 0.1,
        sigma: float = 0.0,
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
            name=f"TT_multihead_M{n_heads}_l{lam_str}_s{sig_str}",
        )
        self._base_tt  = base_tt
        self.n_heads   = n_heads
        self.lambda_div = lambda_div
        self.sigma     = sigma
        self.user_heads: Optional[nn.ModuleList] = None  # set in prepare()

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        # base_tt の ZCA / LogQ / item_head を引き継ぎ
        self._device             = self._base_tt._device
        self.whitener            = self._base_tt.whitener
        self.logq                = self._base_tt.logq
        self.whitened_user_embs  = self._base_tt.whitened_user_embs
        self.whitened_item_embs  = self._base_tt.whitened_item_embs
        self.item_head           = copy.deepcopy(self._base_tt.item_head)
        self._proj_item_embs     = self._base_tt._proj_item_embs

        # M 個の user ヘッドを base_tt.user_head から deepcopy して初期化
        self.user_heads = nn.ModuleList([
            copy.deepcopy(self._base_tt.user_head)
            for _ in range(self.n_heads)
        ]).to(self._device)

        # user_head には list の 0 番目を代入（build_index / recommend の互換性のため）
        self.user_head = self.user_heads[0]

        log.info(
            f"[{self.name}] Fine-tuning {self.n_heads} heads: "
            f"lambda_div={self.lambda_div}, sigma={self.sigma}"
        )
        self._train_multihead(train_pos)

        # item projections は base_tt のまま使う（item_head は変更なし）

    def _train_multihead(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] pairs={len(pairs_t)}, M={self.n_heads}, epochs={self.epochs}")

        # 全 user_heads のパラメータを最適化
        params = list(self.user_heads.parameters())
        opt   = torch.optim.Adam(params, lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)

        self.user_heads.train()
        self.item_head.eval()  # item_head は固定

        # ヘッドのペア組み合わせを事前計算
        from itertools import combinations
        head_pairs = list(combinations(range(self.n_heads), 2))
        n_pairs = max(len(head_pairs), 1)

        # div loss 用アイテムサブセット（計算コスト削減）
        n_div_items = 500
        div_item_idx = torch.randperm(N_items)[:n_div_items]
        X_div = X_item[div_item_idx]  # (n_div_items, 768)
        with torch.no_grad():
            proj_div = self.item_head(X_div)  # (n_div_items, hidden_dim)

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss = 0.0
            n_batches  = 0

            for i in range(0, len(pairs_t), self.batch_size):
                bp    = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                with torch.no_grad():
                    proj_p = self.item_head(X_item[p_idx])
                    proj_n = self.item_head(X_item[n_idx])

                # 各ヘッドの BPR 損失
                bpr_total = torch.tensor(0.0, device=dev)
                queries = []
                for m, head in enumerate(self.user_heads):
                    q_m = head(X_user[u_idx])  # (B, hidden_dim)
                    queries.append(q_m)
                    s_pos = (q_m * proj_p).sum(-1)
                    s_neg = (q_m * proj_n).sum(-1)
                    bpr_total = bpr_total + (-F.logsigmoid(s_pos - s_neg).mean())
                bpr_total = bpr_total / self.n_heads

                # ヘッド間 soft_jaccard 多様性損失
                div_total = torch.tensor(0.0, device=dev)
                if self.lambda_div > 0 and len(head_pairs) > 0:
                    for hi, hj in head_pairs:
                        q_i = queries[hi]   # (B, hidden_dim)
                        q_j = queries[hj]
                        # soft_jaccard: (n_div_items, hidden_dim) @ (hidden_dim, B) → (n_div_items, B)
                        s_i = (proj_div @ q_i.T)    # (n_div_items, B)
                        s_j = (proj_div @ q_j.T)
                        p_i = F.softmax(s_i, dim=0)
                        p_j = F.softmax(s_j, dim=0)
                        inter = torch.min(p_i, p_j).sum(0).mean()
                        union = torch.max(p_i, p_j).sum(0).mean()
                        div_total = div_total + inter / (union + 1e-10)
                    div_total = div_total / n_pairs

                loss = bpr_total + self.lambda_div * div_total

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches  += 1

            sched.step()

            if (epoch + 1) % 10 == 0:
                log.info(
                    f"[{self.name}] epoch={epoch+1}/{self.epochs}  "
                    f"loss={epoch_loss/n_batches:.4f}"
                )

        self.user_heads.eval()
        log.info(f"[{self.name}] Multi-head training complete.")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        head_idx = int(rng.integers(0, self.n_heads))
        dev = self._device
        u_t = torch.from_numpy(
            self.whitened_user_embs[user_idx: user_idx + 1]
        ).float().to(dev)
        with torch.no_grad():
            q = self.user_heads[head_idx](u_t).cpu().numpy()[0]
        if self.sigma > 0:
            noise = rng.normal(0, self.sigma, size=q.shape).astype(np.float32)
            q = q + noise
            norm = np.linalg.norm(q)
            if norm > 1e-9:
                q = q / norm
        return q.astype(np.float32)
