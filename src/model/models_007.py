"""
Plan 007 Models: Two-Tower MLP Head
-------------------------------------
ZCA whitened embeddings (768-dim, frozen) を入力とし、
User/Item それぞれ独立した MLP Head を BPR で学習する Two-Tower モデル。

アーキテクチャ:
    User Tower: whitened_user_emb → MLP_user → L2Norm → proj_user
    Item Tower: whitened_item_emb → MLP_item → L2Norm → proj_item
    スコア = proj_user · proj_item
    損失   = BPR: -log σ(s_pos − s_neg)

Sub-exp:
    7B: depth × hidden_dim スイープ (9条件)
    7C: 最良 Two-Tower + soft_jaccard 多様性アダプタ (5条件)
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

log = logging.getLogger(__name__)


# ── MLP Head ─────────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    """
    Flexible MLP projection head.

    depth=1: Linear(in, out) → L2Norm
    depth=2: Linear(in, h) → LayerNorm → ReLU → Linear(h, out) → L2Norm
    depth=3: Linear(in, h) → LayerNorm → ReLU → Linear(h, h) → LayerNorm → ReLU → Linear(h, out) → L2Norm

    Note: LayerNorm (not BatchNorm) to avoid train/eval discrepancy and batch-size sensitivity.
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 128, depth: int = 2):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        layers: list[nn.Module] = []

        if depth == 1:
            layers.append(nn.Linear(in_dim, hidden_dim))
        else:
            # First layer: in_dim → hidden_dim
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            # Middle layers: hidden_dim → hidden_dim
            for _ in range(depth - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            # Last linear (no activation before L2Norm)
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


# ── Two-Tower Model ───────────────────────────────────────────────────────────

class TwoTowerModel(BaseModel):
    """
    User/Item 独立 MLP Head を BPR で学習する Two-Tower 推薦モデル。

    入力: ZCA whitened embeddings (M0_strong と同じ Whitener を使用)
    出力: proj_user · proj_item で内積スコアリング + LogQ 補正
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        logit_scale: float = 14.3,
        alpha: float = 0.1,
        name: Optional[str] = None,
    ):
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.logit_scale = logit_scale
        self.alpha = alpha
        self.name = name or f"TwoTower_d{depth}_h{hidden_dim}"

        # Set during prepare()
        self.whitener: Optional[Whitener] = None
        self.logq: Optional[LogQCorrector] = None
        self.whitened_user_embs: Optional[np.ndarray] = None
        self.whitened_item_embs: Optional[np.ndarray] = None
        self.user_head: Optional[MLPHead] = None
        self.item_head: Optional[MLPHead] = None
        self._device: Optional[torch.device] = None
        self._proj_item_embs: Optional[np.ndarray] = None  # cached item projections
        self._proj_user_embs: Optional[np.ndarray] = None  # cached user projections

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        super().prepare(train_pos, user_embeddings, item_embeddings, device)
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        n_items = len(item_embeddings)

        # Step 1: ZCA Whitening
        log.info(f"[{self.name}] Fitting ZCA Whitener ...")
        self.whitener = Whitener()
        self.whitener.fit(item_embeddings)
        self.whitened_item_embs = self.whitener.transform(item_embeddings)
        self.whitened_user_embs = self.whitener.transform(user_embeddings)

        # Step 2: Log-Q popularity correction
        self.logq = LogQCorrector(train_pos, n_items, alpha=self.alpha)

        # Step 3: Build and train MLP towers
        self.user_head = MLPHead(768, self.hidden_dim, self.depth).to(self._device)
        self.item_head = MLPHead(768, self.hidden_dim, self.depth).to(self._device)
        self._train(train_pos)

        # Step 4: Pre-compute projected item and user embeddings
        log.info(f"[{self.name}] Pre-computing item & user projections ...")
        self._proj_item_embs = self._project_items()
        self._proj_user_embs = self._project_users()

    def _train(self, train_pos: dict):
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        # Collect (user_idx, pos_item_idx) pairs
        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] Training: pairs={len(pairs_t)}, depth={self.depth}, "
                 f"hidden_dim={self.hidden_dim}, epochs={self.epochs}")

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
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                # Project through independent towers
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
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs}  "
                         f"loss={epoch_loss/n_batches:.4f}")

        self.user_head.eval()
        self.item_head.eval()
        log.info(f"[{self.name}] Training complete.")

    def _project_items(self) -> np.ndarray:
        """全アイテムを item_head で変換し numpy に返す。"""
        dev = self._device
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        self.item_head.eval()
        batch_size = 1024
        results = []
        with torch.no_grad():
            for i in range(0, len(X_item), batch_size):
                proj = self.item_head(X_item[i: i + batch_size])
                results.append(proj.cpu().float().numpy())
        return np.vstack(results)

    def _project_users(self) -> np.ndarray:
        """全ユーザーを user_head で一括変換し numpy に返す。"""
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        self.user_head.eval()
        batch_size = 1024
        results = []
        with torch.no_grad():
            for i in range(0, len(X_user), batch_size):
                proj = self.user_head(X_user[i: i + batch_size])
                results.append(proj.cpu().float().numpy())
        return np.vstack(results)

    def build_index(self) -> faiss.IndexFlatIP:
        """投影済みアイテム埋め込みで FAISS インデックスを構築して返す。"""
        assert self._proj_item_embs is not None
        index = faiss.IndexFlatIP(self._proj_item_embs.shape[1])
        index.add(self._proj_item_embs.astype(np.float32))
        return index

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        """キャッシュされたユーザー投影ベクトルを返す。"""
        if self._proj_user_embs is not None:
            return self._proj_user_embs[user_idx]
        dev = self._device
        u_t = torch.from_numpy(
            self.whitened_user_embs[user_idx: user_idx + 1]
        ).float().to(dev)
        with torch.no_grad():
            proj = self.user_head(u_t)
        return proj.cpu().numpy()[0]

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

        if self.logq is not None:
            penalties = self.logq.get_penalties(cand_indices)
            adjusted = self.logit_scale * cand_raw_scores - self.alpha * penalties
            top_k = np.argsort(-adjusted)[:k]
            return list(cand_indices[top_k])
        else:
            return list(cand_indices[:k])


# ── Two-Tower + Post-training Noise (inference-time) ──────────────────────────

class TwoTowerPostNoise(TwoTowerModel):
    """
    学習済み Two-Tower の推論時に proj_user へガウシアンノイズを加えるバリアント。

    get_query_vector で:
        proj_user = user_head(whitened_user_emb)
        q = L2Norm(proj_user + N(0, sigma))
    """

    def __init__(self, base_tt: TwoTowerModel, sigma: float):
        super().__init__(
            hidden_dim=base_tt.hidden_dim,
            depth=base_tt.depth,
            lr=base_tt.lr,
            epochs=base_tt.epochs,
            batch_size=base_tt.batch_size,
            logit_scale=base_tt.logit_scale,
            alpha=base_tt.alpha,
            name=f"{base_tt.name}_postnoise_s{str(sigma).replace('.','p')}",
        )
        self._base_tt = base_tt
        self.sigma = sigma

    def prepare(self, train_pos, user_embeddings, item_embeddings, device="cuda"):
        # 学習済み base を全て引き継ぎ、再学習なし
        self._device             = self._base_tt._device
        self.whitener            = self._base_tt.whitener
        self.logq                = self._base_tt.logq
        self.whitened_user_embs  = self._base_tt.whitened_user_embs
        self.whitened_item_embs  = self._base_tt.whitened_item_embs
        self.user_head           = self._base_tt.user_head
        self.item_head           = self._base_tt.item_head
        self._proj_item_embs     = self._base_tt._proj_item_embs
        log.info(f"[{self.name}] Using base Two-Tower (no re-training). sigma={self.sigma}")

    def get_query_vector(self, user_idx: int, trial: int, rng: np.random.Generator) -> np.ndarray:
        dev = self._device
        u_t = torch.from_numpy(
            self.whitened_user_embs[user_idx: user_idx + 1]
        ).float().to(dev)
        with torch.no_grad():
            proj = self.user_head(u_t)              # (1, hidden_dim)
        proj_np = proj.cpu().numpy()[0]             # (hidden_dim,)
        noise = rng.normal(0, self.sigma, size=proj_np.shape).astype(np.float32)
        noisy = proj_np + noise
        norm = np.linalg.norm(noisy)
        if norm < 1e-9:
            return proj_np
        return (noisy / norm).astype(np.float32)


# ── Two-Tower + Training-time Noise (noise augmentation) ──────────────────────

class TwoTowerTrainNoise(TwoTowerModel):
    """
    BPR 学習時に whitened user embedding へガウシアンノイズを注入するバリアント。
    ノイズ注入により汎化性能・多様性の両立を狙う。

    学習中:
        u_noisy = whitened_user_emb + N(0, sigma)
        proj_user = user_head(u_noisy)  ← ノイズ入り入力
    推論時 (get_query_vector):
        proj_user = user_head(whitened_user_emb)  ← 決定論的（ノイズなし）
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 2,
        sigma: float = 0.05,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        logit_scale: float = 14.3,
        alpha: float = 0.1,
    ):
        name = f"TwoTower_d{depth}_h{hidden_dim}_trainnoise_s{str(sigma).replace('.','p')}"
        super().__init__(
            hidden_dim=hidden_dim,
            depth=depth,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            logit_scale=logit_scale,
            alpha=alpha,
            name=name,
        )
        self.sigma = sigma

    def _train(self, train_pos: dict):
        """sigma > 0 のノイズを学習中のユーザー入力に加える点のみ TwoTowerModel と異なる。"""
        dev = self._device
        X_user = torch.from_numpy(self.whitened_user_embs).float().to(dev)
        X_item = torch.from_numpy(self.whitened_item_embs).float().to(dev)
        N_items = len(self.whitened_item_embs)

        pairs = []
        for u_idx, pos_items in train_pos.items():
            for it in pos_items:
                pairs.append((int(u_idx), int(it)))
        pairs_t = torch.tensor(pairs, dtype=torch.long)
        log.info(f"[{self.name}] Training with noise augmentation: "
                 f"pairs={len(pairs_t)}, sigma={self.sigma}, "
                 f"depth={self.depth}, hidden_dim={self.hidden_dim}")

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
                n_idx = torch.randint(0, N_items, (len(bp),), device=dev)

                # ノイズを user embedding に加算（学習時のみ）
                u_emb = X_user[u_idx]
                noise = torch.randn_like(u_emb) * self.sigma
                u_emb_noisy = u_emb + noise

                proj_u = self.user_head(u_emb_noisy)   # ノイズ入り
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
                log.info(f"[{self.name}] epoch={epoch+1}/{self.epochs}  "
                         f"loss={epoch_loss/n_batches:.4f}")

        self.user_head.eval()
        self.item_head.eval()
        log.info(f"[{self.name}] Training with noise augmentation complete.")
