"""
Plan 005 Models & Enhancements
------------------------------
Includes:
  - Whitener: ZCA/PCA whitening on item & user embeddings to eliminate anisotropy & hubness.
  - LogQCorrector: Popularity log-frequency correction with CLIP-style logit scaling (1/tau).
  - Models for Sub-exp 5A (Ablation):
      M0_raw, M0_whiten, M0_logq, M0_scaled_logq, M0_strong
  - Models for Sub-exp 5B (Gaussian Noise on Best 5A Model):
      M4_GaussOnModel
"""

from __future__ import annotations

import logging
import numpy as np
import faiss
from src.model.models import BaseModel

log = logging.getLogger(__name__)


# ── Whitening Module ──────────────────────────────────────────────────────────

class Whitener:
    """
    ZCA / PCA Whitening for dense vector embeddings.
    Transforms e' = L2_norm((e - mean) @ W)
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.W: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> Whitener:
        log.info(f"[Whitener] Fitting whitening matrix on embeddings shape={embeddings.shape} ...")
        self.mean = embeddings.mean(axis=0, keepdims=True)
        centered = embeddings - self.mean
        cov = (centered.T @ centered) / len(embeddings)
        
        # SVD of covariance matrix
        U, S, _ = np.linalg.svd(cov)
        inv_sqrt_s = 1.0 / np.sqrt(S + self.eps)
        
        # ZCA Whitening matrix: W = U @ diag(1/sqrt(S)) @ U.T
        self.W = (U * inv_sqrt_s) @ U.T
        log.info("[Whitener] Fitting completed.")
        return self

    def transform(self, vec: np.ndarray) -> np.ndarray:
        if self.mean is None or self.W is None:
            return vec
        is_1d = vec.ndim == 1
        if is_1d:
            vec = vec.reshape(1, -1)
        centered = vec - self.mean
        whitened = centered @ self.W
        norms = np.linalg.norm(whitened, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        res = (whitened / norms).astype(np.float32)
        return res[0] if is_1d else res


# ── Log-Q Popularity Corrector ────────────────────────────────────────────────

class LogQCorrector:
    """
    Log-Q popularity bias correction with optional CLIP logit scaling (1/tau).
    score(u, i) = logit_scale * cos_sim(u, i) - alpha * log(q_i)
    """

    def __init__(self, train_pos: dict[int, np.ndarray], n_total_items: int, alpha: float = 0.1):
        self.alpha = alpha
        counts = np.zeros(n_total_items, dtype=np.float64)
        for item_indices in train_pos.values():
            for iid in item_indices:
                if 0 <= iid < n_total_items:
                    counts[iid] += 1.0
        
        # Laplace smoothing
        prob = (counts + 1.0) / (counts.sum() + n_total_items)
        self.log_q = np.log(prob).astype(np.float32)
        log.info(f"[LogQCorrector] Computed log_q for {n_total_items} items (min={self.log_q.min():.2f}, max={self.log_q.max():.2f}).")

    def get_penalties(self, item_indices: np.ndarray) -> np.ndarray:
        return self.log_q[item_indices]


# ── Sub-exp 5A Models ─────────────────────────────────────────────────────────

class M0_EnhancedBase(BaseModel):
    def __init__(
        self,
        name: str = "M0_enhanced",
        use_whitening: bool = False,
        use_logq: bool = False,
        logit_scale: float = 1.0,
        alpha: float = 0.1,
    ):
        self.name = name
        self.use_whitening = use_whitening
        self.use_logq = use_logq
        self.logit_scale = logit_scale
        self.alpha = alpha
        
        self.whitener: Whitener | None = None
        self.logq: LogQCorrector | None = None
        self.transformed_user_embs: np.ndarray | None = None
        self.transformed_item_embs: np.ndarray | None = None

    def prepare(self, train_pos_items_per_user, user_embeddings, item_embeddings, device="cpu"):
        super().prepare(train_pos_items_per_user, user_embeddings, item_embeddings, device)
        n_items = len(item_embeddings)
        
        if self.use_whitening:
            self.whitener = Whitener()
            self.whitener.fit(item_embeddings)
            self.transformed_item_embs = self.whitener.transform(item_embeddings)
            self.transformed_user_embs = self.whitener.transform(user_embeddings)
        else:
            self.transformed_item_embs = item_embeddings.copy()
            self.transformed_user_embs = user_embeddings.copy()

        if self.use_logq:
            self.logq = LogQCorrector(train_pos_items_per_user, n_items, alpha=self.alpha)

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        return self.transformed_user_embs[user_idx].copy()

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
        scores, I = index.search(q, min(candidate_pool_size, index.ntotal))
        cand_indices = I[0]
        cand_raw_scores = scores[0]

        if self.use_logq and self.logq is not None:
            penalties = self.logq.get_penalties(cand_indices)
            adjusted_scores = (self.logit_scale * cand_raw_scores) - (self.alpha * penalties)
            top_k = np.argsort(-adjusted_scores)[:k]
            return list(cand_indices[top_k])
        else:
            return list(cand_indices[:k])


# ── Sub-exp 5B Model (Gaussian Noise on Enhanced Base Model) ──────────────────

class M4_GaussOnEnhancedModel(M0_EnhancedBase):
    def __init__(
        self,
        base_model: M0_EnhancedBase,
        sigma: float = 0.05,
    ):
        super().__init__(
            name=f"{base_model.name}_gauss_s{sigma:.3f}".replace(".", "p"),
            use_whitening=base_model.use_whitening,
            use_logq=base_model.use_logq,
            logit_scale=base_model.logit_scale,
            alpha=base_model.alpha,
        )
        self.sigma = sigma
        self.base_model = base_model

    def prepare(self, train_pos_items_per_user, user_embeddings, item_embeddings, device="cpu"):
        # Reuse prepared embeddings & objects from base_model if available
        self.transformed_item_embs = self.base_model.transformed_item_embs
        self.transformed_user_embs = self.base_model.transformed_user_embs
        self.whitener = self.base_model.whitener
        self.logq = self.base_model.logq

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        vec = self.transformed_user_embs[user_idx].copy()
        noise = rng.normal(0, self.sigma, size=vec.shape).astype(np.float32)
        return self._normalise(vec + noise)
