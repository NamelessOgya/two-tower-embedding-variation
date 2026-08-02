"""
Plan 006 Models: Diversity Methods Evaluated on Strong Baseline (M0_strong)
-------------------------------------------------------------------------
Base Model (M0_strong):
  - ZCA Whitening on embeddings to eliminate anisotropy & hubness
  - Log-Q popularity correction (alpha=0.1)
  - CLIP-style Logit Auto-Scaling (scale=14.3, tau=0.07)

Diversity Methods:
  - M4_strong: Gaussian Noise (sigma sweep)
  - M5_strong: MC Feature Dropout (p sweep)
  - 3B_strong: Diversity Adapter (lambda sweep across loss functions)
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from src.model.models import BaseModel
from src.model.models_005 import M0_EnhancedBase, Whitener, LogQCorrector
from src.model.models_003 import DIV_LOSSES, NEEDS_ITEMS

log = logging.getLogger(__name__)


def format_lambda_str(lam: float) -> str:
    s = f"{lam:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


# ── M4 Gaussian Noise on M0_strong ───────────────────────────────────────────

class M4_strong_Gauss(M0_EnhancedBase):
    def __init__(self, sigma: float = 0.05, name: str | None = None):
        if name is None:
            name = f"M4_strong_gauss_s{sigma:.3f}".replace(".", "p")
        super().__init__(
            name=name,
            use_whitening=True,
            use_logq=True,
            logit_scale=14.3,
            alpha=0.1,
        )
        self.sigma = sigma

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        vec = self.transformed_user_embs[user_idx].copy()
        noise = rng.normal(0, self.sigma, size=vec.shape).astype(np.float32)
        return self._normalise(vec + noise)


# ── M5 MC Dropout on M0_strong ───────────────────────────────────────────────

class M5_strong_Dropout(M0_EnhancedBase):
    def __init__(self, dropout_rate: float = 0.2, name: str | None = None):
        if name is None:
            name = f"M5_strong_dropout_p{dropout_rate:.2f}".replace(".", "p")
        super().__init__(
            name=name,
            use_whitening=True,
            use_logq=True,
            logit_scale=14.3,
            alpha=0.1,
        )
        self.dropout_rate = dropout_rate

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        vec = self.transformed_user_embs[user_idx].copy()
        mask = (rng.random(size=vec.shape) > self.dropout_rate).astype(np.float32)
        scale = 1.0 / max(1.0 - self.dropout_rate, 1e-9)
        vec = vec * mask * scale
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return self.transformed_user_embs[user_idx].copy()
        return (vec / norm).astype(np.float32)


# ── 3B Diversity Adapter on M0_strong ────────────────────────────────────────

class _DivAdapterModule(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.full((dim,), -3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(self.log_sigma)
        eps = torch.randn_like(x)
        noisy = x + sigma * eps
        return F.normalize(noisy, p=2, dim=-1)


class M3B_strong_Adapter(M0_EnhancedBase):
    """
    3B Diversity Adapter trained on whitened user embeddings,
    evaluated with M0_strong Log-Q & CLIP logit scaling re-ranking.
    """

    def __init__(
        self,
        div_loss_name: str,
        lambda_div: float = 1.0,
        lr: float = 2e-3,
        epochs: int = 50,
        batch_size: int = 512,
    ):
        lam_str = format_lambda_str(lambda_div)
        name = f"3B_strong_{div_loss_name}_l{lam_str}"
        super().__init__(
            name=name,
            use_whitening=True,
            use_logq=True,
            logit_scale=14.3,
            alpha=0.1,
        )
        self.div_loss_name = div_loss_name
        self.div_fn = DIV_LOSSES[div_loss_name]
        self.lambda_div = lambda_div
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.adapter_module: _DivAdapterModule | None = None
        self._device: torch.device | None = None

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        super().prepare(train_pos, user_embeddings, item_embeddings, device)
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._train_adapter(train_pos)

    def _train_adapter(self, train_pos):
        dev = self._device
        X_user = torch.from_numpy(self.transformed_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.transformed_item_embs).float().to(dev)
        N_items = len(self.transformed_item_embs)
        needs_items = self.div_loss_name in NEEDS_ITEMS

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items[:30]:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] Training adapter: pairs={len(pairs_t)}, λ={self.lambda_div}")

        self.adapter_module = _DivAdapterModule(dim=768).to(dev)
        opt = torch.optim.Adam(self.adapter_module.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)

        for epoch in range(self.epochs):
            perm = torch.randperm(len(pairs_t))
            for i in range(0, len(pairs_t), self.batch_size):
                bp = pairs_t[perm[i: i + self.batch_size]].to(dev)
                u_idx = bp[:, 0]
                p_idx = bp[:, 1]
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                u_emb = X_user[u_idx]
                p_emb = X_item[p_idx]
                n_emb = X_item[n_idx]

                q1 = self.adapter_module(u_emb)
                q2 = self.adapter_module(u_emb)

                s1_pos = (q1 * p_emb).sum(-1)
                s1_neg = (q1 * n_emb).sum(-1)
                s2_pos = (q2 * p_emb).sum(-1)
                s2_neg = (q2 * n_emb).sum(-1)

                bpr1 = -F.logsigmoid(s1_pos - s1_neg).mean()
                bpr2 = -F.logsigmoid(s2_pos - s2_neg).mean()
                bpr_loss = bpr1 + bpr2

                if needs_items:
                    d_loss = self.div_fn(q1, q2, X_item)
                else:
                    d_loss = self.div_fn(q1, q2)

                loss = bpr_loss + self.lambda_div * d_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

            sched.step()

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        if self.adapter_module is None:
            return self.transformed_user_embs[user_idx].copy()
        dev = self._device
        self.adapter_module.eval()
        with torch.no_grad():
            u_t = torch.from_numpy(self.transformed_user_embs[user_idx: user_idx + 1]).float().to(dev)
            q_t = self.adapter_module(u_t)
            return q_t.cpu().numpy()[0]
