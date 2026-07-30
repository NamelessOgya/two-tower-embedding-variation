"""
M0–M6 Two-Tower Diversity Models
---------------------------------
All models:
  - Accept precomputed L2-normalised user/item embeddings (768-dim)
  - Return query vector(s) without updating the item ANN index
  - item embeddings are FIXED throughout the experiment

Models:
  M0  Baseline           – deterministic, same vector every trial
  M1  Clustering         – K-means on user's positive training items
  M2  RandomAttention    – Dirichlet-weighted combo of training items
  M3  RandomSubset       – randomly picks from pre-generated text variants
  M4  GaussianNoise      – user_emb + N(0,σ²), renormalised
  M5  MCDropout          – random feature masking, renormalised
  M6  VAE                – trained VAE head; sample z = μ + σ·ε per trial
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

log = logging.getLogger(__name__)


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseModel:
    name: str = "base"

    def prepare(
        self,
        train_pos_items_per_user: dict[int, np.ndarray],
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        device: str = "cpu",
    ) -> None:
        """Store shared references; subclasses may train/precompute here."""
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

    def get_query_vector(
        self, user_idx: int, trial: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return a single L2-normalised query vector of shape (768,)."""
        raise NotImplementedError

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)


# ── M0 – Baseline ─────────────────────────────────────────────────────────────

class M0_Baseline(BaseModel):
    name = "M0_baseline"

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        return self.user_embeddings[user_idx].copy()


# ── M1 – Clustering ───────────────────────────────────────────────────────────

class M1_Clustering(BaseModel):
    name = "M1_clustering"

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters

    def prepare(self, train_pos_items_per_user, user_embeddings, item_embeddings, device="cpu"):
        super().prepare(train_pos_items_per_user, user_embeddings, item_embeddings, device)
        log.info(f"[M1] Building cluster centroids (k={self.n_clusters}) ...")
        self.centroids: dict[int, np.ndarray] = {}

        for user_idx, item_indices in train_pos_items_per_user.items():
            items = item_embeddings[item_indices].astype(np.float32)
            k = min(self.n_clusters, len(items))
            if k <= 1:
                self.centroids[user_idx] = items.copy()
                continue
            km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
            km.fit(items)
            centers = km.cluster_centers_.astype(np.float32)
            centers = np.array([self._normalise(c) for c in centers])
            self.centroids[user_idx] = centers

        log.info(f"[M1] Done. {len(self.centroids)} users have centroids.")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        centroids = self.centroids.get(user_idx)
        if centroids is None or len(centroids) == 0:
            return self.user_embeddings[user_idx].copy()
        idx = rng.integers(0, len(centroids))
        return centroids[idx].copy()


# ── M2 – Random Attention ─────────────────────────────────────────────────────

class M2_RandomAttention(BaseModel):
    """Dirichlet-weighted soft combination of user's positive training items."""
    name = "M2_random_attention"

    def prepare(self, train_pos_items_per_user, user_embeddings, item_embeddings, device="cpu"):
        super().prepare(train_pos_items_per_user, user_embeddings, item_embeddings, device)
        self.user_history = train_pos_items_per_user  # user_idx -> item_indices

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        item_indices = self.user_history.get(user_idx)
        if item_indices is None or len(item_indices) == 0:
            return self.user_embeddings[user_idx].copy()
        items = self.item_embeddings[item_indices].astype(np.float32)
        weights = rng.dirichlet(np.ones(len(items)))
        vec = (weights[:, None] * items).sum(axis=0)
        return self._normalise(vec)


# ── M3 – Random Attribute Subset ──────────────────────────────────────────────

class M3_RandomSubset(BaseModel):
    """Randomly selects from pre-generated user text variant embeddings."""
    name = "M3_random_subset"

    def __init__(self, variants_path: str):
        self.variants_path = Path(variants_path)

    def prepare(self, train_pos_items_per_user, user_embeddings, item_embeddings, device="cpu"):
        super().prepare(train_pos_items_per_user, user_embeddings, item_embeddings, device)
        log.info(f"[M3] Loading variants from {self.variants_path}")
        self.variants = np.load(self.variants_path)  # (N_users, N_variants, 768)
        log.info(f"[M3] variants shape: {self.variants.shape}")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        n_variants = self.variants.shape[1]
        v_idx = rng.integers(0, n_variants)
        return self.variants[user_idx, v_idx].copy()


# ── M4 – Gaussian Noise ───────────────────────────────────────────────────────

class M4_GaussianNoise(BaseModel):
    name = "M4_gaussian_noise"

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        vec = self.user_embeddings[user_idx].copy()
        noise = rng.normal(0, self.sigma, size=vec.shape).astype(np.float32)
        return self._normalise(vec + noise)


# ── M5 – MC Dropout ───────────────────────────────────────────────────────────

class M5_MCDropout(BaseModel):
    """Random feature masking (inverted dropout) on user embedding."""
    name = "M5_mc_dropout"

    def __init__(self, dropout_rate: float = 0.2):
        self.dropout_rate = dropout_rate

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        vec = self.user_embeddings[user_idx].copy()
        mask = (rng.random(size=vec.shape) > self.dropout_rate).astype(np.float32)
        # inverted dropout scaling
        scale = 1.0 / max(1.0 - self.dropout_rate, 1e-9)
        vec = vec * mask * scale
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return self.user_embeddings[user_idx].copy()
        return (vec / norm).astype(np.float32)


# ── M6 – VAE ──────────────────────────────────────────────────────────────────

class _VAEHead(nn.Module):
    def __init__(self, input_dim: int = 768, latent_dim: int = 128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_lv = nn.Linear(256, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
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
        return self.dec(z), mu, lv


class M6_VAE(BaseModel):
    name = "M6_vae"

    def __init__(
        self,
        latent_dim: int = 128,
        beta: float = 1.0,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
    ):
        self.latent_dim = latent_dim
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.vae: _VAEHead | None = None
        self.mu_cache: np.ndarray | None = None
        self.lv_cache: np.ndarray | None = None
        self._device: torch.device | None = None

    def prepare(self, train_pos_items_per_user, user_embeddings, item_embeddings, device="cpu"):
        super().prepare(train_pos_items_per_user, user_embeddings, item_embeddings, device)
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info(f"[M6] Training VAE on {len(user_embeddings)} user embeddings "
                 f"(device={self._device}, epochs={self.epochs}) ...")

        self.vae = _VAEHead(input_dim=768, latent_dim=self.latent_dim).to(self._device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr)

        X = torch.from_numpy(user_embeddings).float().to(self._device)
        n = len(X)

        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self._device)
            total_loss, n_batches = 0.0, 0
            for i in range(0, n, self.batch_size):
                x = X[perm[i: i + self.batch_size]]
                recon, mu, lv = self.vae(x)
                recon_loss = F.mse_loss(recon, x)
                kl_loss = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = recon_loss + self.beta * kl_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 25 == 0:
                log.info(f"[M6]   epoch {epoch+1}/{self.epochs}  "
                         f"loss={total_loss/n_batches:.4f}")

        # Cache μ and log-σ² for all users
        self.vae.eval()
        with torch.no_grad():
            mu_all, lv_all = self.vae.encode(X)
        self.mu_cache = mu_all.cpu().numpy()
        self.lv_cache = lv_all.cpu().numpy()
        log.info(f"[M6] Training done. latent shape={self.mu_cache.shape}")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.mu_cache[user_idx]
        sigma = np.exp(0.5 * self.lv_cache[user_idx])
        eps = rng.standard_normal(size=mu.shape).astype(np.float32)
        z = torch.from_numpy((mu + sigma * eps)[None]).float().to(self._device)
        with torch.no_grad():
            recon = self.vae.decode(z).squeeze(0).cpu().numpy()
        return self._normalise(recon)
