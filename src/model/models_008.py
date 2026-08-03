"""
Plan 008 Models: Two-Tower + Diversity Loss / Noise Position Variants
----------------------------------------------------------------------
Plan 007 の最良 Two-Tower (depth=2, hidden=64) をベースに、
多様性制御手法を強化した追加モデルを定義する。

Sub-exp:
    8B: TwoTowerDivLoss  - BPR + diversity loss を統合学習（6損失 × λ sweep）
                           推論時も入力ノイズで多様化
    8C: TwoTowerInputNoiseBoth  - MLP 入力前ノイズ（学習時・推論時）
        TwoTowerOutputNoiseBoth - MLP 出力後ノイズ（学習時・推論時）
"""

from __future__ import annotations

import copy
import logging
from typing import Callable

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.models import BaseModel
from src.model.models_005 import Whitener, LogQCorrector
from src.model.models_007 import TwoTowerModel, MLPHead
from src.model.models_003 import DIV_LOSSES, NEEDS_ITEMS

log = logging.getLogger(__name__)


# ── Sub-exp 8B: Two-Tower + Diversity Loss (end-to-end joint training) ────────

class TwoTowerDivLoss(TwoTowerModel):
    """
    Plan 003 3B の Two-Tower 版。BPR + diversity loss を統合学習。

    最良 Two-Tower の MLP 重みを deepcopy し、BPR + div_loss で fine-tune する。
    user_head のみ更新し、item_head は固定（計算コスト削減）。

    学習:
        q1 = user_head(whitened + sigma*eps1),  eps1 ~ N(0,I)
        q2 = user_head(whitened + sigma*eps2),  eps2 ~ N(0,I)  (独立サンプル)
        L = BPR(q1) + BPR(q2) + lambda_div * div_loss(q1, q2)
    推論:
        q = user_head(whitened + sigma*eps)  <- trial ごとに異なる eps
    """

    def __init__(
        self,
        base_tt: TwoTowerModel,
        div_loss_name: str,
        lambda_div: float = 0.5,
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
            name=f"TT_divloss_{div_loss_name}_l{lam_str}_s{sig_str}",
        )
        self._base_tt = base_tt
        self.div_loss_name = div_loss_name
        self._div_fn: Callable = DIV_LOSSES[div_loss_name]
        self._needs_items: bool = div_loss_name in NEEDS_ITEMS
        self.lambda_div = lambda_div
        self.sigma = sigma

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        # base_tt の ZCA / LogQ を引き継ぎ（再計算不要）
        self._device            = self._base_tt._device
        self.whitener           = self._base_tt.whitener
        self.logq               = self._base_tt.logq
        self.whitened_user_embs = self._base_tt.whitened_user_embs
        self.whitened_item_embs = self._base_tt.whitened_item_embs
        self._proj_item_embs    = self._base_tt._proj_item_embs  # fine-tune 前キャッシュ

        # MLP 重みを deepcopy して fine-tune（base_tt を破壊しない）
        self.user_head = copy.deepcopy(self._base_tt.user_head)
        self.item_head = copy.deepcopy(self._base_tt.item_head)

        log.info(f"[{self.name}] Fine-tuning: loss={self.div_loss_name}, "
                 f"lambda={self.lambda_div}, sigma={self.sigma}")
        self._train_divloss(train_pos)

        # user_head は更新しているが item_head は固定なので proj_item は変わらない
        # ただし念のため再計算
        self._proj_item_embs = self._project_items()

    def _train_divloss(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_proj_item = torch.from_numpy(self._proj_item_embs).float().to(dev)
        N_items = len(self._proj_item_embs)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items[:30]:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)

        opt = torch.optim.Adam(
            list(self.user_head.parameters()),  # user_head のみ更新
            lr=self.lr
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=1e-5
        )
        self.user_head.train()
        self.item_head.eval()

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss, n_batches = 0.0, 0

            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                u_emb = X_user[u_idx]
                # 独立した 2 つのノイズで stochastic query を生成
                q1 = self.user_head(u_emb + torch.randn_like(u_emb) * self.sigma)
                q2 = self.user_head(u_emb + torch.randn_like(u_emb) * self.sigma)

                proj_p = X_proj_item[p_idx]
                proj_n = X_proj_item[n_idx]

                bpr = (-F.logsigmoid((q1 * proj_p).sum(-1) - (q1 * proj_n).sum(-1))
                       - F.logsigmoid((q2 * proj_p).sum(-1) - (q2 * proj_n).sum(-1))).mean()

                if self._needs_items:
                    d_loss = self._div_fn(q1, q2, X_proj_item)
                else:
                    d_loss = self._div_fn(q1, q2)

                loss = bpr + self.lambda_div * d_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1

            sched.step()

            if (epoch + 1) % 10 == 0:
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs}  "
                         f"loss={epoch_loss/n_batches:.4f}")

        self.user_head.eval()
        log.info(f"[{self.name}] Fine-tuning complete.")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        """推論時も入力ノイズで多様化（trial ごとに異なる eps）。"""
        dev = self._device
        u_t = torch.from_numpy(
            self.whitened_user_embs[user_idx: user_idx + 1]
        ).float().to(dev)
        noise_np = rng.normal(
            0, self.sigma, size=(1, self.whitened_user_embs.shape[1])
        ).astype(np.float32)
        noise = torch.from_numpy(noise_np).to(dev)
        with torch.no_grad():
            proj = self.user_head(u_t + noise)
        return proj.cpu().numpy()[0]


# ── Sub-exp 8C: TrainNoise 位置バリアント × 推論時ノイズあり ─────────────────

class TwoTowerInputNoiseBoth(TwoTowerModel):
    """
    whitened embedding (MLP 入力) へのノイズを学習時・推論時の両方で付与。

    学習: whitened + N(0, sigma) -> MLP -> L2Norm -> proj_user
    推論: whitened + N(0, sigma) -> MLP -> L2Norm -> proj_user  (trial ごとに異なる)

    - Plan 007E の TwoTowerTrainNoise との違い: 推論時にもノイズを付与する
    - Plan 007D の TwoTowerPostNoise との違い: ノイズが MLP の非線形変換を通る
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        depth: int = 2,
        sigma: float = 0.05,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        logit_scale: float = 14.3,
        alpha: float = 0.1,
    ):
        name = f"TT_inputnoise_both_s{str(sigma).replace('.', 'p')}"
        super().__init__(
            hidden_dim=hidden_dim, depth=depth, lr=lr, epochs=epochs,
            batch_size=batch_size, logit_scale=logit_scale, alpha=alpha, name=name,
        )
        self.sigma = sigma

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
        log.info(f"[{self.name}] Training (input noise, train+infer): sigma={self.sigma}")

        opt = torch.optim.Adam(
            list(self.user_head.parameters()) + list(self.item_head.parameters()), lr=self.lr
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)
        self.user_head.train()
        self.item_head.train()

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss, n_batches = 0.0, 0
            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx, p_idx = bp[:, 0], bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)
                u_emb = X_user[u_idx]
                proj_u = self.user_head(u_emb + torch.randn_like(u_emb) * self.sigma)
                proj_p = self.item_head(X_item[p_idx])
                proj_n = self.item_head(X_item[n_idx])
                loss = -F.logsigmoid(
                    (proj_u * proj_p).sum(-1) - (proj_u * proj_n).sum(-1)
                ).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            sched.step()
            if (epoch + 1) % 10 == 0:
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs}  "
                         f"loss={epoch_loss/n_batches:.4f}")
        self.user_head.eval()
        self.item_head.eval()
        log.info(f"[{self.name}] Training complete.")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        """推論時も入力ノイズを付与（trial ごとに異なる eps）。"""
        dev = self._device
        u_t = torch.from_numpy(
            self.whitened_user_embs[user_idx: user_idx + 1]
        ).float().to(dev)
        noise_np = rng.normal(
            0, self.sigma, size=(1, self.whitened_user_embs.shape[1])
        ).astype(np.float32)
        noise = torch.from_numpy(noise_np).to(dev)
        with torch.no_grad():
            proj = self.user_head(u_t + noise)
        return proj.cpu().numpy()[0]


class TwoTowerOutputNoiseBoth(TwoTowerModel):
    """
    MLP 出力 (proj_user) へのノイズを学習時・推論時の両方で付与するバリアント。

    学習: MLP(whitened) + N(0, sigma) -> L2Norm -> proj_user  (BPR)
    推論: MLP(whitened) + N(0, sigma) -> L2Norm -> proj_user  (trial ごとに異なる)

    - Post-Noise (Plan 007D) との違い: MLP が「ノイズ付き出力で positive を予測する」
      ように学習されるため、ノイズ方向への汎化能力を持つ。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        depth: int = 2,
        sigma: float = 0.05,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        logit_scale: float = 14.3,
        alpha: float = 0.1,
    ):
        name = f"TT_outputnoise_both_s{str(sigma).replace('.', 'p')}"
        super().__init__(
            hidden_dim=hidden_dim, depth=depth, lr=lr, epochs=epochs,
            batch_size=batch_size, logit_scale=logit_scale, alpha=alpha, name=name,
        )
        self.sigma = sigma

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
        log.info(f"[{self.name}] Training (output noise, train+infer): sigma={self.sigma}")

        opt = torch.optim.Adam(
            list(self.user_head.parameters()) + list(self.item_head.parameters()), lr=self.lr
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)
        self.user_head.train()
        self.item_head.train()

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            epoch_loss, n_batches = 0.0, 0
            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx, p_idx = bp[:, 0], bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)
                # MLP 出力にノイズを加算してから L2Norm
                proj_u_det = self.user_head(X_user[u_idx])
                noise = torch.randn_like(proj_u_det) * self.sigma
                proj_u = F.normalize(proj_u_det + noise, p=2, dim=-1)
                proj_p = self.item_head(X_item[p_idx])
                proj_n = self.item_head(X_item[n_idx])
                loss = -F.logsigmoid(
                    (proj_u * proj_p).sum(-1) - (proj_u * proj_n).sum(-1)
                ).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            sched.step()
            if (epoch + 1) % 10 == 0:
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs}  "
                         f"loss={epoch_loss/n_batches:.4f}")
        self.user_head.eval()
        self.item_head.eval()
        log.info(f"[{self.name}] Training complete.")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        """推論時も MLP 出力にノイズを付与（trial ごとに異なる）。"""
        dev = self._device
        u_t = torch.from_numpy(
            self.whitened_user_embs[user_idx: user_idx + 1]
        ).float().to(dev)
        with torch.no_grad():
            proj_det = self.user_head(u_t)
        proj_np = proj_det.cpu().numpy()[0]
        noise = rng.normal(0, self.sigma, size=proj_np.shape).astype(np.float32)
        noisy = proj_np + noise
        norm = np.linalg.norm(noisy)
        return (noisy / norm).astype(np.float32) if norm > 1e-9 else proj_np
