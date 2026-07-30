"""
Plan 002 モデル群
-----------------
M4_sweep     : σ を複数スイープする Gaussian Noise
M6a/b/c/d   : VAE 改良版 (β-VAE + Cosine loss)
M5a          : Dropout rate スイープ
M5b          : Structured dropout (group-level mask)
M5c          : Soft dropout (mask → noise 置換)
"""

from __future__ import annotations

import logging
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.models import BaseModel

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Sub-exp 2A: M4 Gaussian σ スイープ
# ══════════════════════════════════════════════════════════════════════════════

class M4_GaussianNoise_Sweep(BaseModel):
    """Gaussian Noise with configurable σ (for sweep experiments)."""

    def __init__(self, sigma: float):
        self.sigma = sigma
        self.name = f"M4_gauss_sigma{sigma:.4f}".replace(".", "p")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        vec = self.user_embeddings[user_idx].copy()
        noise = rng.normal(0, self.sigma, size=vec.shape).astype(np.float32)
        return self._normalise(vec + noise)


SIGMA_VALUES = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def make_gaussian_sweep_models() -> list[M4_GaussianNoise_Sweep]:
    return [M4_GaussianNoise_Sweep(s) for s in SIGMA_VALUES]


# ══════════════════════════════════════════════════════════════════════════════
# Sub-exp 2B: M6 VAE 改良版
# ══════════════════════════════════════════════════════════════════════════════

class _ImprovedVAEHead(nn.Module):
    def __init__(self, input_dim: int = 768, latent_dim: int = 128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_lv = nn.Linear(256, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.decode(z), mu, lv


def _cosine_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity (both inputs L2-normalised inside)."""
    recon_n = F.normalize(recon, p=2, dim=-1)
    target_n = F.normalize(target, p=2, dim=-1)
    return (1.0 - (recon_n * target_n).sum(dim=-1)).mean()


class M6_VAE_Improved(BaseModel):
    """
    Improved VAE with:
      - GELU activations (smoother gradients)
      - Configurable β (β-VAE: large β resolves posterior collapse)
      - Configurable loss_fn: 'mse' | 'cosine'
    """

    def __init__(
        self,
        name_tag: str,
        latent_dim: int = 128,
        beta: float = 1.0,
        loss_fn: str = "mse",   # "mse" or "cosine"
        lr: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 256,
    ):
        self.name = f"M6_{name_tag}"
        self.latent_dim = latent_dim
        self.beta = beta
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.vae: _ImprovedVAEHead | None = None
        self.mu_cache: np.ndarray | None = None
        self.lv_cache: np.ndarray | None = None
        self._device: torch.device | None = None

    def prepare(self, train_pos_items_per_user, user_embeddings, item_embeddings, device="cpu"):
        super().prepare(train_pos_items_per_user, user_embeddings, item_embeddings, device)
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info(f"[{self.name}] Training VAE: β={self.beta}, loss={self.loss_fn}, "
                 f"epochs={self.epochs}, device={self._device}")

        self.vae = _ImprovedVAEHead(input_dim=768, latent_dim=self.latent_dim).to(self._device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        X = torch.from_numpy(user_embeddings).float().to(self._device)
        n = len(X)

        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self._device)
            total_loss, n_batches = 0.0, 0
            for i in range(0, n, self.batch_size):
                x = X[perm[i: i + self.batch_size]]
                recon, mu, lv = self.vae(x)

                if self.loss_fn == "cosine":
                    recon_loss = _cosine_loss(recon, x)
                else:
                    recon_loss = F.mse_loss(recon, x)

                kl_loss = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = recon_loss + self.beta * kl_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()
            if (epoch + 1) % 50 == 0:
                log.info(f"[{self.name}]   epoch {epoch+1}/{self.epochs}  "
                         f"loss={total_loss/n_batches:.5f}")

        self.vae.eval()
        with torch.no_grad():
            mu_all, lv_all = self.vae.encode(X)
        self.mu_cache = mu_all.cpu().numpy()
        self.lv_cache = lv_all.cpu().numpy()

        # 診断: latent σの平均（これが大きいほど posterior collapse が解消されている）
        mean_sigma = np.exp(0.5 * self.lv_cache).mean()
        log.info(f"[{self.name}] Done. latent μ norm={np.linalg.norm(self.mu_cache, axis=1).mean():.3f}, "
                 f"mean σ={mean_sigma:.4f}")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.mu_cache[user_idx]
        sigma = np.exp(0.5 * self.lv_cache[user_idx])
        eps = rng.standard_normal(size=mu.shape).astype(np.float32)
        z = torch.from_numpy((mu + sigma * eps)[None]).float().to(self._device)
        with torch.no_grad():
            recon = self.vae.decode(z).squeeze(0).cpu().numpy()
        return self._normalise(recon)


def make_vae_improved_models() -> list[M6_VAE_Improved]:
    configs = [
        ("beta4_mse",     4.0,  "mse"),
        ("beta8_mse",     8.0,  "mse"),
        ("beta1_cosine",  1.0,  "cosine"),
        ("beta4_cosine",  4.0,  "cosine"),
    ]
    return [M6_VAE_Improved(tag, beta=b, loss_fn=l) for tag, b, l in configs]


# ══════════════════════════════════════════════════════════════════════════════
# Sub-exp 2C: M5 Dropout 改良版
# ══════════════════════════════════════════════════════════════════════════════

class M5_RateSweep(BaseModel):
    """M5 with configurable dropout rate."""

    def __init__(self, dropout_rate: float):
        self.dropout_rate = dropout_rate
        self.name = f"M5_dropout_p{dropout_rate:.2f}".replace(".", "p")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        vec = self.user_embeddings[user_idx].copy()
        mask = (rng.random(size=vec.shape) > self.dropout_rate).astype(np.float32)
        scale = 1.0 / max(1.0 - self.dropout_rate, 1e-9)
        vec = vec * mask * scale
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 1e-9 else self.user_embeddings[user_idx].copy()


DROPOUT_RATES = [0.05, 0.1, 0.2, 0.3, 0.5]


def make_dropout_sweep_models() -> list[M5_RateSweep]:
    return [M5_RateSweep(p) for p in DROPOUT_RATES]


class M5_StructuredDropout(BaseModel):
    """
    Group-level dropout: 768次元を n_groups グループに分割し、
    グループ単位でマスクする。次元内の相関が保たれ精度低下が緩和されることを期待。
    """
    name = "M5_structured"

    def __init__(self, n_groups: int = 12, group_dropout_rate: float = 0.2):
        self.n_groups = n_groups          # 768 / 12 = 64次元 per group
        self.group_dropout_rate = group_dropout_rate
        self.name = f"M5_structured_g{n_groups}_p{group_dropout_rate:.2f}".replace(".", "p")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        vec = self.user_embeddings[user_idx].copy()  # (768,)
        dim = len(vec)
        group_size = dim // self.n_groups
        mask = np.ones(dim, dtype=np.float32)
        for g in range(self.n_groups):
            if rng.random() < self.group_dropout_rate:
                start = g * group_size
                end = start + group_size
                mask[start:end] = 0.0
        scale = 1.0 / max(1.0 - self.group_dropout_rate, 1e-9)
        vec = vec * mask * scale
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 1e-9 else self.user_embeddings[user_idx].copy()


class M5_SoftDropout(BaseModel):
    """
    Soft dropout: マスクされた次元をゼロにするのではなく
    Gaussian noise で置換する。信号を保ちながら部分探索。

    gate ~ Bernoulli(p)
    q = user_emb * (1-gate) + (user_emb + ε) * gate
    """

    def __init__(self, noise_rate: float = 0.2, sigma: float = 0.05):
        self.noise_rate = noise_rate
        self.sigma = sigma
        self.name = f"M5_soft_p{noise_rate:.2f}_s{sigma:.3f}".replace(".", "p")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        vec = self.user_embeddings[user_idx].copy()
        gate = (rng.random(size=vec.shape) < self.noise_rate).astype(np.float32)
        noise = rng.normal(0, self.sigma, size=vec.shape).astype(np.float32)
        q = vec * (1.0 - gate) + (vec + noise) * gate
        return self._normalise(q)
