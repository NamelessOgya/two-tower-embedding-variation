"""
Plan 018 Models: Two-Tower Training-Time Popularity De-biasing and Anti-Hubness
-------------------------------------------------------------------------------
Two-Tower モデルの学習時（Training Time）において、人気アイテム偏重や Hubness
（幾何学的空間崩壊）を直接抑制する 4 つの学習手法。

Models:
  1. TwoTowerBPRBase: 標準 Two-Tower (一様ランダム負例 + BPR 損失)
  2. TwoTowerLogQInfoNCE: 学習損失内での Log-Q 補正 InfoNCE (Google RecSys'19)
  3. TwoTowerPopNegativeBPR: 人気度偏重ネガティブサンプリング BPR (Google WWW'20)
  4. TwoTowerUniformityLoss: 超球面 Uniformity 正則化 Two-Tower (ICML'20 / KDD'22)
  5. TwoTowerAdaptiveTauInfoNCE: 人気度適応型動的温度 InfoNCE
"""

from __future__ import annotations

import logging
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.models import BaseModel
from src.model.models_005 import Whitener, LogQCorrector
from src.model.models_007 import MLPHead, TwoTowerModel

log = logging.getLogger(__name__)


# ── 1. Base Model (Alias for clarity) ─────────────────────────────────────────

class TwoTowerBPRBase(TwoTowerModel):
    """標準 Two-Tower (一様ランダム負例 + BPR 損失)"""
    def __init__(self, hidden_dim: int = 64, depth: int = 2, lr: float = 1e-3, epochs: int = 50, batch_size: int = 1024, logit_scale: float = 14.3, alpha: float = 0.1, name: Optional[str] = None):
        super().__init__(
            hidden_dim=hidden_dim,
            depth=depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=logit_scale,
            alpha=alpha,
            name=name or f"TT_BPR_base_d{depth}_h{hidden_dim}",
        )


# ── 2. Log-Q In-Batch InfoNCE Two-Tower ───────────────────────────────────────

class TwoTowerLogQInfoNCE(TwoTowerModel):
    """
    Google (RecSys 2019) スタイルの Log-Q 補正 In-Batch InfoNCE 損失。
    損失の分子・分母でアイテム対数頻度 log(q_i) を差し引き、人気アイテムへの過剰勾配を学習時に直接補正。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        depth: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        tau: float = 0.07,
        alpha: float = 1.0,  # 訓練損失内の log-q 引算係数
        eval_alpha: float = 0.1,  # 推論時の log-q 係数
        name: Optional[str] = None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            depth=depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=1.0 / tau,
            alpha=eval_alpha,
            name=name or f"TT_LogQ_InfoNCE_tau{tau}_a{alpha}".replace(".", "p"),
        )
        self.tau = tau
        self.train_alpha = alpha

    def _train(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        # 事前計算された log_q
        assert self.logq is not None
        log_q_t = torch.from_numpy(self.logq.log_q).float().to(dev)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] Training LogQ-InfoNCE: pairs={len(pairs_t)}, tau={self.tau}, alpha={self.train_alpha}")

        opt = torch.optim.Adam(
            list(self.user_head.parameters()) + list(self.item_head.parameters()),
            lr=self.lr
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=1e-5
        )

        self.user_head.train()
        self.item_head.train()

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                B = len(bp)
                if B < 2:
                    continue

                proj_u = self.user_head(X_user[u_idx])      # (B, dim)
                proj_p = self.item_head(X_item[p_idx])      # (B, dim)

                # In-batch 類似度行列 (B, B)
                sim_matrix = torch.matmul(proj_u, proj_p.T) / self.tau  # (B, B)

                # Log-Q 補正: 各アイテム列から alpha * log(q_j) を引く
                item_logq = log_q_t[p_idx].unsqueeze(0)  # (1, B)
                logits = sim_matrix - self.train_alpha * item_logq  # (B, B)

                # 対角成分 (i, i) が正例ターゲット
                targets = torch.arange(B, device=dev, dtype=torch.long)
                loss = F.cross_entropy(logits, targets)

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            sched.step()
            if (epoch + 1) % 10 == 0:
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs}  loss={epoch_loss/max(1, n_batches):.4f}")

        self.user_head.eval()
        self.item_head.eval()
        log.info(f"[{self.name}] Training complete.")


# ── 3. Popularity-Biased Negative Sampling Two-Tower ──────────────────────────

class TwoTowerPopNegativeBPR(TwoTowerModel):
    """
    Google (WWW 2020) スタイルの人気度偏重ネガティブサンプリング (Mixed Negative Sampling)。
    出現頻度 P_neg(j) ∝ q_j^beta (beta=0.5〜0.75) で負例を抽出。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        depth: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        beta: float = 0.75,
        logit_scale: float = 14.3,
        alpha: float = 0.1,
        name: Optional[str] = None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            depth=depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=logit_scale,
            alpha=alpha,
            name=name or f"TT_PopNeg_b{beta}".replace(".", "p"),
        )
        self.beta = beta
        self.neg_probs: Optional[np.ndarray] = None

    def _train(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        # アイテム頻度に基づくサンプリング分布 P(j) ∝ count_j^beta
        counts = np.zeros(N_items, dtype=np.float64)
        for item_indices in train_pos.values():
            for iid in item_indices:
                if 0 <= iid < N_items:
                    counts[iid] += 1.0
        prob = (counts + 1.0) ** self.beta
        prob /= prob.sum()
        self.neg_probs = prob
        prob_t = torch.from_numpy(prob).float().to(dev)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] Training PopNeg BPR: pairs={len(pairs_t)}, beta={self.beta}")

        opt = torch.optim.Adam(
            list(self.user_head.parameters()) + list(self.item_head.parameters()),
            lr=self.lr
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=1e-5
        )

        self.user_head.train()
        self.item_head.train()

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                B = len(bp)

                # 人気度に基づいた多項分布から負例を高速サンプリング
                n_idx = torch.multinomial(prob_t, B, replacement=True)

                proj_u = self.user_head(X_user[u_idx])
                proj_p = self.item_head(X_item[p_idx])
                proj_n = self.item_head(X_item[n_idx])

                s_pos = (proj_u * proj_p).sum(-1)
                s_neg = (proj_u * proj_n).sum(-1)
                loss = -F.logsigmoid(s_pos - s_neg).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            sched.step()
            if (epoch + 1) % 10 == 0:
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs}  loss={epoch_loss/max(1, n_batches):.4f}")

        self.user_head.eval()
        self.item_head.eval()
        log.info(f"[{self.name}] Training complete.")


# ── 4. Hypersphere Uniformity Loss Two-Tower ──────────────────────────────────

class TwoTowerUniformityLoss(TwoTowerModel):
    """
    ICML 2020 (Wang & Isola) / DirectAU (KDD 2022) スタイルの超球面 Uniformity 正則化。
    アイテム埋め込み空間全体の等方分散を最大化し、幾何学的 Hubness 空間崩壊を直接防止。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        depth: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        lambda_uni: float = 1.0,
        t_uni: float = 2.0,
        logit_scale: float = 14.3,
        alpha: float = 0.1,
        name: Optional[str] = None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            depth=depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=logit_scale,
            alpha=alpha,
            name=name or f"TT_Uniformity_l{lambda_uni}".replace(".", "p"),
        )
        self.lambda_uni = lambda_uni
        self.t_uni = t_uni

    def _train(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] Training Uniformity TwoTower: pairs={len(pairs_t)}, lambda_uni={self.lambda_uni}")

        opt = torch.optim.Adam(
            list(self.user_head.parameters()) + list(self.item_head.parameters()),
            lr=self.lr
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=1e-5
        )

        self.user_head.train()
        self.item_head.train()

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss = 0.0
            epoch_uni = 0.0
            n_batches = 0

            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                B = len(bp)
                n_idx = torch.randint(0, N_items, (B,), device=dev)

                proj_u = self.user_head(X_user[u_idx])
                proj_p = self.item_head(X_item[p_idx])
                proj_n = self.item_head(X_item[n_idx])

                # 1. BPR 損失
                s_pos = (proj_u * proj_p).sum(-1)
                s_neg = (proj_u * proj_n).sum(-1)
                bpr_loss = -F.logsigmoid(s_pos - s_neg).mean()

                # 2. アイテム埋め込みの Uniformity 損失 (Wang & Isola, ICML 2020)
                # L_uni = log E_{i,j} [ exp(-t * ||x_i - x_j||^2) ]
                # バッチ内アイテム proj_p の pairwise 二乗ユークリッド距離
                # ||x_i - x_j||^2 = 2 - 2 (x_i · x_j) (L2正規化済みベクトルの場合)
                cos_sim_items = torch.matmul(proj_p, proj_p.T)  # (B, B)
                sq_dist = 2.0 - 2.0 * cos_sim_items
                uni_loss = torch.log(torch.mean(torch.exp(-self.t_uni * sq_dist)) + 1e-8)

                loss = bpr_loss + self.lambda_uni * uni_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                epoch_uni += uni_loss.item()
                n_batches += 1

            sched.step()
            if (epoch + 1) % 10 == 0:
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs}  "
                         f"loss={epoch_loss/max(1, n_batches):.4f}  "
                         f"uni={epoch_uni/max(1, n_batches):.4f}")

        self.user_head.eval()
        self.item_head.eval()
        log.info(f"[{self.name}] Training complete.")


# ── 5. Popularity-Adaptive Temperature InfoNCE Two-Tower ──────────────────────

class TwoTowerAdaptiveTauInfoNCE(TwoTowerModel):
    """
    アイテムの人気度に応じて温度 tau_i を動的変化させる InfoNCE 損失。
    人気アイテム: tau_i が高くマイルドな勾配
    テールアイテム: tau_i が低くシャープで強い勾配
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        depth: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        tau_base: float = 0.07,
        gamma: float = 1.0,  # 温度変動幅
        eval_alpha: float = 0.1,
        name: Optional[str] = None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            depth=depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=1.0 / tau_base,
            alpha=eval_alpha,
            name=name or f"TT_AdaptTau_t{tau_base}_g{gamma}".replace(".", "p"),
        )
        self.tau_base = tau_base
        self.gamma = gamma

    def _train(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        assert self.logq is not None
        log_q = self.logq.log_q
        min_lq, max_lq = log_q.min(), log_q.max()
        norm_lq = (log_q - min_lq) / max(max_lq - min_lq, 1e-6)  # [0, 1]

        # tau_i = tau_base * (1 + gamma * norm_lq)  (人気アイテムほど tau が大)
        item_taus = self.tau_base * (1.0 + self.gamma * norm_lq)
        item_taus_t = torch.from_numpy(item_taus).float().to(dev)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] Training AdaptiveTau InfoNCE: pairs={len(pairs_t)}, tau_base={self.tau_base}, gamma={self.gamma}")

        opt = torch.optim.Adam(
            list(self.user_head.parameters()) + list(self.item_head.parameters()),
            lr=self.lr
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=1e-5
        )

        self.user_head.train()
        self.item_head.train()

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                B = len(bp)
                if B < 2:
                    continue

                proj_u = self.user_head(X_user[u_idx])
                proj_p = self.item_head(X_item[p_idx])

                # (B, B) 類似度行列
                sim_matrix = torch.matmul(proj_u, proj_p.T)  # (B, B)

                # 各アイテム列の動的温度 tau_j で除算
                batch_taus = item_taus_t[p_idx].unsqueeze(0)  # (1, B)
                logits = sim_matrix / batch_taus             # (B, B)

                targets = torch.arange(B, device=dev, dtype=torch.long)
                loss = F.cross_entropy(logits, targets)

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            sched.step()
            if (epoch + 1) % 10 == 0:
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs}  loss={epoch_loss/max(1, n_batches):.4f}")

        self.user_head.eval()
        self.item_head.eval()
        log.info(f"[{self.name}] Training complete.")
